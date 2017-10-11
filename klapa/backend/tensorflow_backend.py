import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables

from collections import defaultdict

import numpy as np
import os

from .common import floatx, epsilon
from .common import image_data_format
from ..utils.generic_utils import has_arg

from .common import set_image_dim_ordering
from .common import image_dim_ordering

py_all = all
py_sum = sum

_SESSION = None

_GRAPH_LEARNING_PHASE = {}

_GRAPH_UID_DICTS = {}

_MANUAL_VAR_INIT = False

def get_uid(prefix=''):
	"""Get the uid of the default graph

	# Arguments:
	prefix: Graph prefix

	# Returns:
	Unique graph identifier
	"""
	global _GRAPH_UID_DICTS
	graph = tf.get_default_graph()
	if graph not in _GRAPH_UID_DICTS:
		_GRAPH_UID_DICTS[graph] = defaultdict(int)
	_GRAPH_UID_DICTS[graph][prefix] += 1
	return _GRAPH_UID_DICTS[graph][prefix]

def reset_uids():
	global _GRAPH_UID_DICTS
	_GRAPH_UID_DICTS = {}

def clear_session():
	"""Destroy current TF graph and create new one
		Get rid of clutter from old models/layers
	"""
	global _SESSION
	global _GRAPH_LEARNING_PHASE
	tf.reset_default_graph()
	reset_uids()
	_SESSION = None
	phase = tf.placeholder(dtype='bool', name='klapa_learning_phase')
	_GRAPH_LEARNING_PHASE = {}
	_GRAPH_LEARNING_PHASE[tf.get_default_graph()] = phase

def manual_variable_initialization(value):
	"""Set the manual variable initialization flag.
	Set if variables need to be initialized (default) or
	if user should handle the initialization
	"""
	global _MANUAL_VAR_INIT
	_MANUAL_VAR_INIT = value

def learning_phase():
	"""Return the learning phase flag
	The learning phase flag is a bool tensor and is passed
	as input to Klapa function that uses different behavior at
	train and test time
	"""
	graph = tf.get_default_graph()
	if graph not in _GRAPH_LEARNING_PHASE:
		phase = tf.placeholder(dtype='bool',
							   name='klapa_learning_phase')
		_GRAPH_LEARNING_PHASE[graph] = phase
		return _GRAPH_LEARNING_PHASE[graph]

def set_learning_phase(value):
	"""Sets the learning phase to a fixed value
	Argument:
	value: Learning phase value, 0 or 1
	Raises:
	ValueError if 'value' is neither '0' or '1'
	"""
	global _GRAPH_LEARNING_PHASE
	if value not in {0,1}:
		raise ValueError('Expected learning phase to be'
						 '0 or 1')
		_GRAPH_LEARNING_PHASE[tf.get_default_graph()] = value

def get_session():
	""" Return the tensorflow session to be used in backend
	If TF session is avaiable, it will be returned else
	global Klapa session is returned

	If no Klapa session is available, this will create a new
	global session
	"""
	global _SESSION
	if tf.get_default_session() is not None:
		session = tf.get_default_session()
	else:
		if _SESSION is None:
			if not os.environ.get('OMP_NUM_THREADS'):
				config = tf.ConfigProto(allow_soft_placement=True)
			else:
				num_thread = int(os.environ.get('OMP_NUM_THREADS'))
				config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
										allow_soft_placement=True)
			_SESSION = tf.Session(config=config)
		session = _SESSION

	if not _MANUAL_VAR_INIT:
		with session.graph.as_default():
			variables = tf.global_variables()
			candidate_vars = []
			for v in variables:
				if not getattr(v, '_klapa_initialized', False):
					candidate_vars.append(v)

			is_initialized = session.run(
				[tf.is_variable_initialized(v) for v in candidate_vars])
			uninitialized_vars = []
			for flag, v in zip(is_initialized, candidate_vars):
				if not flag:
					uninitialized_vars.append(v)
				v._klapa_initialized = True
			if uninitialized_vars:
				session.run(tf.variables_initializer(uninitialized_vars))
	return session

def set_session(session):
	global _SESSION
	_SESSION = session

def _to_tensor(x, dtype):
	x = tf.convert_to_tensor(x)
	if x.dtype != dtype:
		x = tf.cast(x, dtype)
	return x

def is_sparse(tensor):
	"""Return true if the tensor is sparse
	"""
	return isinstance(tensor, tf.SparseTensor)

def to_dense(tensor):
	"""Convert sparse tensor to dense tensor
	"""
	if is_sparse(tensor):
		return tf.sparse_tensor_to_dense(tensor)
	else:
		return tensor

name_scope = tf.name_scope

def variable(value, dtype=None, name=None, constraint=None):
	if dtype is None:
		dtype = floatx()
	if hasattr(value, 'tocoo'):
		sparse_coo = value.tocoo()
		indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
								  np.expand_dims(sparse_coo.col, 1)), 1)
		v = tf.SparseTensor(indices=indices,
							values=sparse_coo.data,
							dense_shape=sparse_coo.shape)
		v._klapa_shape = sparse_coo.shape
		v._uses_learning_phase = False
		return v

	v = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
	if isinstance(value, np.ndarray):
		v._klapa_shape = value.shape
	elif hasattr(value, 'get_shape'):
		v._keras_shape = int_shape(value)
	v._uses_learning_phase = False

	try:
		v.constraint = constraint
	except AttributeError:
		v._constraint = constraint
	return v

def is_klapa_tensor(x):
	"""Return true if x is klapa tensor
	from klapa import backend as K
	from klapa.layers import Input, Dense
	np_val = numpy.array([1, 2])
	K.is_klapa_tensor(np_val)
		ValueError # A numpy array is not symbolic tensor
	k_var = tf.placeholder('float32', shape=(1,1))
	K.is_klapa_tensor(k_var)
		False # A variable created outside klapa is not klapa tensor
	klapa_var = K.variable(np_var)
	K.is_klapa_tensor(klapa_var)
		False # A variable created with klapa backend is not klapa tensor
	klapa_placeholder = K.placeholder(shape=(2,4,5))
	K.is_klapa_tensor(klapa_placeholder)
		False # A placeholder is not a klapa tensor
	klapa_input = Input([10])
	K.is_keras_tensor(keras_input)
		True # An Input is a klapa tensor
	klapa_layer_output = Dense(10)(klapa_input)
	K.is_klapa_tensor(klapa_layer_output)
		True # Any klapa layer is tensor
	"""
	if not isinstance(x, (tf.Tensor, tf_variables.Variable, 
					  tf.SparseTensor)):
		raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) + '`. '
						 'Expected a symbolic tensor instance.')
	return hasattr(x, '_keras_history')

def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
	"""Instantiate a placeholder tensor and return it

	Example:
	from klapa import backend as K
	input_ph = K.placeholder(shape=(2, 4, 5))
	input_ph._klapa_shape
		(2, 4, 5)
	input_ph
	<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
	"""
	if dtype is None:
		dtype = floatx()
	if not shape:
		if ndim:
			shape = tuple([None for _ in range(ndim)])
	if sparse:
		x = tf.sparse_placeholder(dtype, shape=shape, name=name)
	x._keras_shape = shape
	x._uses_learning_phase = False
	return x

def is_placeholder(x):
	try:
		return x.op.type == 'Placeholder'
	except AttributeError:
		return False

def shape(x):
	"""Return symbolic shape of the tensor
	Example:
	from klapa import backend as K
	tf_session = K.get_session()
	val = np.array([1, 2], [3, 4])
	kval = K.variable(value=val)
	inputs = klapa.backend.placeholder(shape=(2,4,5))
	K.shape(kvar)
	<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
	K.shape(inputs)
	K.shape(kval).eval(session=tf_session)
	K..shape(inputs).eval(session=tf_session)
	"""
	return tf.shape(x)





































