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

def int_shape(x):
	"""Returns the shape tensor or variable as a tuple of int or None entries
	Example:
	from klapa import backend as K
	inputs = K.placeholder(shape=(2,4,5))
	K.int_shape(inputs)
	val = np.array([[1,2], [3 4]])
	kvar = K.variable(value = val)
	K.int_shape(kvar)
	"""
	if(hasattr(x, '_keras_shape')):
		return x._keras_shape
	try:
		return tuple(x.get_shape().as_list())
	except ValueError:
		return None

def ndim(x):
	"""Return number of axes on the tensor
	Example:
	from klapa import backend as K
	inputs = K.placeholder(shape=(2,4,5))
	val = np.array([[1,2], [3,4]])
	kvar = K.variable(value=val)
	K.ndim(inputs)
	K.ndim(kvar)
	"""
	dims = x.get_shape()._dims
	if dims is not None:
		return len(dims)
	return None

def dtype(x):
	"""Return the dtype of a klapa tensor or variable as string

	Example:
	from klapa import backend as K
	K.dtype(K.placeholder(shape=(2,4,5)))
	"""
	return x.dtype.base_dtype.name

def eval(x):
	"""Evaluate the value of the variable
	Example:
	from klapa import backend as K
	kvar = K.variable(np.array([[1,2], [3,4], dtype='float32']))
	K.eval(kvar)
	"""
	return to_dense(x).eval(session=get_session())

def zeros(shape, dtype=None, name=None):
	"""Instantiate all-zero variable and return
	Example:
	from klapa import Backend as K
	kvar = K.zeros((3,4))
	K.eval(kvar)
	"""
	if dtype is None:
		dtype = floatx()
	tf_dtype = tf.as_dtype(dtype)
	return variable(tf.constant_initializer(0., dtype=tf_dtype)(shape), dtype, name)

def ones(shape, dtype=None, name=None):
	"""Instantiate all-ones tensor variable and return
	Example:
	from klapa import backend as K
	kvar = K.ones((3,4))
	K.eval(kvar)
	"""
	if dtype is None:
		dtype = floatx()
	tf_dtype = tf.as_dtype(dtype)
	return variable(tf.constant_initializer(1., dtype=tf_dtype)(shape), dtype, name)

def eyes(size, dtype=None, name=None):
	"""Instantiate an identity matrix and return it
	Example:
	from klapa import backend as K
	kvar = K.eye(3)
	K.eval(kvar)
	"""
	return variable(np.eye(size), dtype, name)

def zeros_like(x, dtype=None, name=None):
	"""Instantiate all zero variable of the same shape as another tensor
	"""
	return tf.zeros_like(x, dtype=dtype, name=name)

def ones_like(x, dtype=None, name=None):
	return tf.ones_like(x, dtype=dtype, name=None)

def identity(x):
	"""Return a tensor with the same content as input
	"""
	return tf.identity(x)

def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
	"""Instantiate a variable with values drawn from unform distribution

	Example:
	kvar = K.random_uniform_variable((2,3), 0, 1)
	K.eval(kvar)
	"""
	if dtype is None:
		dtype = floatx()
	tf_dtype = tf.as_dtype(dtype)
	if seed is None:
		seed = np.random.randint(10e8)
	value = tf.random_uniform_initializer(low, high, dtype=dtype, seed=seed)(shape)
	return variable(value, dtype=dtype, name=name)

def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
	"""Instantiate a variable with values drawn from normal distribution

	Example:
	kvar = K.random_normal_variable((2,3), 0, 1)
	K.eval(kvar)
	"""
	if dtype is None:
		dtype = floatx()
	tf_dtype = tf.as_dtype(dtype)
	if seed is None:
		seed = np.random.randint(10e8)
	value = tf.random_normal_initializer(mean, scale, dtype=tf_dtype, seed=seed)(shape)
	return variable(value, dtype=dtype, name=name)

def count_params(x):
	"""Return number of scalars in klapa variable
	Example:
	kvar = K.zeros((2,3))
	K.count_params(kvar)
	"""
	shape = x.get_shape()
	return np.prod([shape[i]._value for i in range(len(shape))])

def cast(x, dtype):
	"""Casts a tensor to different dtype and return it
	Example:
	from klapa import Backend as K
	input = K.placeholder((2,3), dtype='float32')
	input = K.cast(input, dtype='float16')
	"""
	return tf.cast(x, dtype)

def update(x, new_x):
	"""udpate the value of x to new_x
	"""
	return tf.assign(x, new_x)

def update_add(x, increment):
	"""update the value of x by adding increment
	"""
	return tf.assign_add(x, increment)

def update_sub(x, decrement):
	"""Update the value of x by subtracting decrementing
	"""
	return tf.assign_sub(x, decrement)

def moving_average_update(x, value, momentum):
	"""compute the moving average of a variable
	"""
	return moving_averages.assign_moving_average(x, value, momentum, zero_debias=False)

# LINEAR ALGEBRA
def dot(x, y):
	"""Muultiply two tensors and return value
	Example:
	x = K.placeholder(shape=(2, 3))
	y = K.placeholder(shape=(3, 4))
	xy = K.dot(x, y)

	With nD tensor multiplication, the behavior is like Theano
	"""
	if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
		x_shape = []
		for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
			if i is not None:
				x_shape.append(i)
			else:
				x_shape.append(s)
		x_shape = tuple(x_shape)
		y_shape = []
		for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
			if i is not None:
				y_shape.append(i)
			else:
				y_shape.append(s)

		y_shape = tuple(y_shape)
		y_permute_dim = list(range(ndim(y)))
		y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
		xt = tf.reshape(x, [-1, x_shape[-1]])
		yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
		return tf.reshape(tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])

	if is_sparse(x):
		out = tf.sparse_tensor_dense_matmul(x, y)
	else:
		out = tf.matmul(x, y)
	return out

def batch_dot(x, y, axes=None):
	"""Batchwise dot product
	batch_dot results in tensor or variable with less dimensions than 
	input.
	input is klapa tensor or variable with ndim >=2
	Example:
	x = [[1, 2], [3, 4]] and y = [[5, 6], [7, 8]]
	batch_dot(x, y, axes=1) = [[17, 53]]

	x_batch = K.ones(shape=(32, 20, 1))
	y_batch = K.ones(shape=(32, 30, 20))
	xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1,2])
	K.int_shape(xy_batch_dot)
	O/P: (32, 1, 30)
	"""
	if isinstance(axes, int):
		axes = (axes, axes)
	x_ndim = ndim(x)
	y_ndim = ndim(y)
	if x_ndim > y_ndim:
		diff = x_ndim - y_ndim
		y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
	elif y_ndim > x_ndim:
		diff = y_ndim - x_ndim
		x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
	else:
		diff = 0
	if ndim(x) == 2 and ndim(y) == 2:
		if axes[0] == axes[1]:
			out = tf.reduce_sum(tf.multiply(x, y), axes[0])
		else:
			out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1,0]), y), axes[1])
	else:
		if axes is not None:
			adj_x = None if axes[0] == ndim(x) - 1 else True
			adj_y = True if axes[1] == ndim(y) - 1 else None
		else:
			adj_x = None
			adj_y = None
		out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
	if diff:
		if x_ndim > y_ndim:
			idx = x_ndim + y_ndim - 3
		else:
			idx = x_ndim - 1
		out = tf.squeeze(out, list(range(idx, idx + diff)))
	if ndim(out) == 1:
		out = expand_dims(out, 1)
	return out

def transpose(x):
	"""Transpose a tensor
	"""
	return tf.transpose(x)

def gather(reference, indices):
	"""retrieve teh elements of indices in a tensor reference
	"""
	return tf.gather(reference, indices)

# ELEMENT WISE OPERATIONS

def max(x, axis=None, keepdims=False):
	"""Maximum value in a tensor
	"""
	return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

def min(x, axis=None, keepdims=False):
	"""minimum value in a tensor
	"""
	return tf.reduce_min(x, axis=axis, keep_dims=keepdims)

def sum(x, axis=None, keepdims=False):
	"""Sum the values in a tensor along side a specific axis
	"""
	return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

def prod(x, axis=None, keepdims=False):
	"""Multiplies the value in the tensor alongside an axis
	"""
	return tf.reduce_prod(x, axis=axis, keep_dims=keepdims)

def cumsum(x, axis=0):
	"""Cumulative sum of the values of tensor, alongside a specific axis
	"""
	return tf.cumsum(x, axis=axis)

def cumprod(x, axis=0):
	"""cumulative product of the value in tensor alongside an axis
	"""
	return tf.cumprod(x, axis=axis)

def var(x, axis=None, keepdims=False):
	"""Variance of a tensor alongside an axis
	"""
	if x.dtype.base_dtype == tf.bool:
		x = tf.cast(x, floatx())
	m = tf.reduce_mean(x, axis=axis, keep_dims=True)
	devs_squared = tf.square(x - m)
	return tf.reduce_mean(dev_squared, axis=axis, keep_dims=keepdims)

def std(x, axis=None, keepdims=False):
	"""Standard deviation of the tensor, alongside the specific axis
	"""
	return tf.sqrt(var(x, axis=axis, keepdims=keepdims))

def mean(x, axis=None, keepdims=False):
	if x.dtype.base_dtype == tf.bool:
		x = tf.cast(x, floatx())
	return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)

def any(x, axis=None, keepdims=False):
	"""Logical OR"""
	x = tf.cast(x, tf.bool)
	return tf.reduce_any(x, axis=axis, keep_dims=keepdims)

def all(x, axis=None, keepdims=False):
	"""Logical AND"""
	x = tf.cast(x, tf.bool)
	return tf.reduce_all(x, axis=axis, keep_dims=keepdims)

def argmax(x, axis=-1):
	"""Return the index of max value along the axis
	"""
	return tf.argmax(x, axis)

def argmin(x, axis=-1):
	"""Return the index of minimum value along an axis
	"""
	return tf.argmin(x, axis)

def square(x):
	return tf.square(x)

def abs(x):
	return tf.abs(x)

def sqrt(x):
	zero = _to_tensor(0., x.dtype.base_dtype)
	inf = _to_tensor(np.inf, x.dtype.base_dtype)
	x = tf.clip_by_value(x, zero, inf)
	return tf.sqrt(x)

def exp(x):
	return tf.exp(x)

def log(x):
	return tf.log(x)

def logsumexp(x, axis=None, keepdims=False):
	"""Compute log(sum(exp(elements across a dim in tensor)))
	This implementation is better than log(sum(exp(x))) because it 
	protects from large input overflows and small input underflows
	"""
	return tf.reduce_logsumexp(x, axis=axis, keep_dims=keepdims)

def round(x):
	"""Element wise rounding to closest integer
	"""
	return tf.round(x)

def sign(x):
	return tf.sign(x)

def pow(x, a):
	return tf.pow(x, a)

def clip(x, min_value, max_value):
	if(max_value is not None and max_value < min_value):
		max_value = min_value
	if max_value is None:
		max_value = np.inf
	min_value = _to_tensor(min_value, x.dtype.base_dtype)
	max_value = _to_tensor(max_value, x.dtype.base_dtype)
	return tf.clip_by_value(x, min_value, max_value)

def equal(x, y):
	return tf.equal(x, y)

def not_equal(x, y):
	return tf.not_equal(x, y)

def greater(x, y):
	return tf.greater(x, y)

def greater_equal(x, y):
	return tf.greater_equal(x, y)

def less(x, y):
	return tf.less(x, y)

def less_equal(x, y):
	return tf.less_equal(x, y)

def maximum(x, y):
	return tf.maximum(x, y)

def minimum(x, y):
	return tf.minimum(x, y)

def sin(x):
	return tf.sin(x)

def cos(x):
	return tf.cos(x)

def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
	"""Compute mean and std for a batch then apply batch_normalization on batch
	"""
	mean, var = tf.nn.moments(x, reduction_axes, shift=None, name=None, keep_dims=False)
	if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
	else:
		target_shape = []
		for axis in range(ndim(x)):
			if axis in reduction_axes:
				target_shape.append(1)
			else:
				target_shape.append(tf.shape(x)[axis])
		target_shape = tf.stack(target_shape)

		broadcast_mean = tf.reshape(mean, target_shape)
		broadcast_var = tf.reshape(var, target_shape)
		if gamma is None:
			broadcast_gamma = None
		else:
			broadcast_gamma = tf.reshape(gamma, target_shape)
		if beta is None:
			broadcast_beta = None
		else:
			broadcast_beta = tf.reshape(beta, target_shape)
		normed = tf.nn.batch_normalization(x, broadcast_mean, broadcast_var, broadcast_beta,
											broadcast_gamma, epsilon)
	returned normed, mean, var

def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
	"""Applies batch normalization on x given mean, var, beta and gamma
	"""
	return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
	"""Concatenate a list of tensors alongside the specific axis
	"""
	if axis < 0:
		rank = ndim(tensors[0])
		if rank:
			axis %= rank
		else:
			axis = 0

	if py_all([is_sparse(x) for x in tensors]):
		return tf.sparse_concat(axis, tensors)
	else:
		return tf.concat([to_dense(x) for x in tensors], axis)

def reshape(x, shape):
	return tf.reshape(x, shape)

def permute_dimension(x, pattern):
	return tf.transpose(x, perm=pattern)

def resize_images(x, height_factor, width_factor, data_format):
	"""Resize the image in 4D tensor
	"""
	if data_format == 'channels_first':
		original_shape = int_shape(x)
		new_shape = tf.shape(x)[2:]
		new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
		x = permute_dimension(x, [0, 2, 3, 1])
		x = tf.image.resize_nearest_neighbor(x, new_shape)
		x = permute_dimension(x, [0, 3, 1, 2])
		x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None, 
						original_shape[3] * width_factor if original_shape[3] is not None else None))
		return x
	elif data_format == 'channels_last':
		original_shape = int_shape(x)
		new_shape = tf.shape(x)[1:3]
		new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
		x = tf.image.resize_nearest_neighbor(x, new_shape)
		x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
					original_shape[2] * width_factor if original_shape[2] is not None else None))
		return x
	else:
		raise ValueError('Invalid data format:', data_format)

def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
	"""Resize volume contained in 5D tensor
	"""
	if data_format == 'channels_first':
		output = repeat_elements(x, depth_factor, axis=2)
		output = repeat_elements(output, height_factor, axis=3)
		output = repeat_elements(output, width_factor, axis=4)
		return output
	elif data_format == 'channels_last':
		output = repeat_elements(x, depth_factor, axis=1)
		output = repeat_elements(output, height_factor, axis=2)
		output = repeat_elements(output, width_factor, axis=3)
		return output
	else:
		raise ValueError('Invalid data_format:', data_format)

def repeat_elements(x, rep, axis):
	"""Repeats the elements of a tensor along axis. If x has shape (s1, s2, s3) and axis is 1,
	the output will be (s1, s2*rep, s3)"""
	x_shape = x.get_shape().as_list()
	if x_shape[axis] is not None:
		splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
		x_rep = [s for s in splits for _ in range(rep)]
		return concatenate(x_rep, axis)

	# x_shape[axis] is None	
	# Repeating
	auxiliary_axis = axis + 1
	x_shape = tf.shape(x)
	x_rep = tf.expand_dims(x, axis=auxiliary_axis)
	reps = np.ones(len(x.get_shape()) + 1)
	reps[auxiliary_axis] = rep
	x_rep = tf.tile(x_rep, reps)

	# Merging
	reps = np.delete(reps, auxiliary_axis)
	reps[axis] = rep
	reps = tf.constant(reps, dtype='int32')
	x_shape = x_shape * reps
	x_rep = tf.reshape(x_rep, x_shape)

	# Fix shape representation
	x_shape = x.get_shape().as_list()
	x_rep.set_shape(x_shape)
	x_rep._keras_shape = tuple(x_shape)
	return x_rep

def repeat(x, n):
	"""Repeat a 2D tensor. If x has shape (samples, dim) and n is 2, the output is (samples, 2, dim)
	"""
	assert ndim(x) == 2:
	x = tf.expand_dims(x, 1)
	pattern = tf.stack([1, n, 1])
	return tf.tile(x, pattern)

def arange(start, stop=None, step=1, dtype='int32'):
	"""Create 1D tensor containing a sequence of integers
	If only one argument is provided, it's a stop argument
	"""
	if stop is None and start < 0:
		start = 0
	result = tf.range(start, limit=stop, delta=step, name='arange')
	if dtype != 'int32':
		result = cast(result, dtype)
	return result

def tile(x, n):
	"""create a tensor by tiling x by n
	"""
	if isinstance(n, int):
		n = [n]
	return tf.tile(x, n)

def flatten(x):
	return tf.reshape(x, [-1])

def batch_flatten(x):
	"""Turn a nD tensor into a 2D tensor with same 0th dimension 
	Flattens each data sample of a batch
	"""
	x = tf.reshape(x, tf.stack([-1, prod(shape(x)[1:])]))

def expand_dims(x, axis=-1):
	"""Add 1 sized dimension at index axis
	"""
	return tf.expand_dims(x, axis)

def squeeze(x, axis):
	"""Removes a 1 dimension from the tensor at index axis
	"""
	return tf.squeeze(x, [axis])

def temporal_padding(x, padding=(1, 1)):
	"""Pad the middle dimension of a 3D array
	"""
	assert len(padding) == 2
	pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
	return tf.pad(x, pattern)

def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
	"""Pads the 2nd and 3rd dimensions of a 4D tensor
	"""
	assert len(padding) == 2
	assert len(padding[0]) == 2
	assert len(padding[1]) == 2

	if data_format is None:
		data_format = image_data_format()
	if data_format not in {'channel_first', 'channel_last'}:
		raise ValueError('Unknown data_format ' + str(data_format))

	if data_format == 'channel_first':
		pattern = [[0, 0],
				   [0, 0],
				   list(padding[0]),
				   list(padding[1])]
	else:
		pattern = [[0, 0],
					list(padding[0]), list(padding[1]),
					[0, 0]]
	return tf.pad(x, pattern)

def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
	"""Pads 5D tensor with zeros along with depth height and width dimension
	Pads these dimensions with respectively padding[0], padding[1] and
	padding[2] zeros left and right
	"""
	assert len(padding) == 3
	assert len(padding[0]) == 2
	assert len(padding[1]) == 2
	assert len(padding[2]) == 2
	if data_format is None:
		data_format = image_data_format()
	if data_format not in {'channels_last', 'channel_first'}:
		raise ValueError('Unknown data_format: '+str(data_format))

	if data_format == 'channel_first':
		pattern = [
					[0, 0],
					[0, 0],
					[padding[0][0], padding[0][1]],
					[padding[1][0], padding[1][1]],
					[padding[2][0], padding[2][1]]
					]
	else:
		pattern = [
					[0, 0],
					[padding[0][0], padding[0][1]],
					[padding[1][0], padding[1][1]],
					[padding[2][0], padding[2][1]],
					[0, 0]
					]
	return tf.pad(x, pattern)

def stack(x, axis=0):
	"""Stacks a list of rank R tensors into rank R+1 tensor
	"""
	return tf.stack(x, axis=axis)

def one_hot(indices, num_classes):
	return tf.one_hot(indices, depth=num_classes, axis=-1)

def reverse(x, axes):
	if isinstance(axes, int):
		axes = [axes]
	return tf.reverse(x, axes)

# VALUE MANIPULATION

def get_value(x):
	return x.eval(session=get_session())

def batch_get_value(ops):
	"""Returns the value of more than one tensor variable
	"""
	if ops:
		return get_session().run(ops)
	else:
		return []

def set_value(x, value):
	"""Set the value of a variable, from a numpy array
	"""
	value = np.asarray(value, dtype=dtype(x))
	tf_dtype = tf.as_dtype(x.dtype.name.split('_')[0])
	if hasattr(x, '_assign_placeholder'):
		_assign_placeholder = x._assign_placeholder
		assign_op = x._assign_op
	else:
		assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
		assign_op = x.assign(assign_placeholder)
		x._assign_placeholder = assign_placeholder
		x._assign_op = assign_op
	get_session().run(assign_op, feed_dict={assign_placeholder: value})

def batch_set_value(tuples):
	"""Sets the values of many tensor variables at once
	"""
	if tuples:
		assign_ops = []
		feed_dict = {}
		for x, value in tuples:
			value = np.asarray(value, dtype=dtype(x))
			tf_dtype = tf.as_dtype(x.dtype.name.split('_')[0])
			if hasattr(x, '_assign_placeholder'):
				assign_placeholder = x._assign_placeholder
				assign_op = x._assign_op
			else:
				assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
				assign_op = x.assign(assign_placeholder)
				x._assign_placeholder = assign_placeholder
				x._assign_op = assign_op
			assign_ops.append(assign_op)
			feed_dict[assign_placeholder] = value
		get_session().run(assign_ops, feed_dict=feed_dict)

def get_variable_shape(x):
	return int_shape(x)

def print_tensor(x, message=''):
	"""Print message and tensor value when evaluated. 
	"""
	return tf.Print(x, [x], message)

# GRAPH MANIPULATION

class Function(object):
	"""Runs a computation graph
	"""
	def __init__(self, inputs, outputs, updates=None, name=None, **session_kwargs):
		updates = updates or []
		if not isinstance(inputs, (list, tuple)):
			raise TypeError('input to Tensorflow backend function should be list or tuple')
		if not isinstance(outputs, (list, tuple)):
			raise TypeError('output to Tensorflow backend function should be list or tuple')
		if not isinstance(updates, (list, tuple)):
			raise TypeError('update to Tensorflow backend function should be list or tuple')
		self.inputs = list(inputs)
		self.outputs = list(outputs)
		with tf.control_dependencies(self.outputs):
			update_ops = []
			for update in updates:
				if isinstance(update, tuple):
					p, new_p = update
					updates_ops.append(tf.assign(p, new_p))
				else:
					updates_ops.append(update)
			self.updates_op = tf.group(*updates_ops)
		self.name = name
		self.session_kwargs = session_kwargs

	def __call__(self, inputs):
		if not isinstance(inputs, (list, tuple)):
			raise ValueError('Input should be a list or tuple')
		feed_dict = {}
		for tensor, value in zip(self.inputs, inputs):
			if is_sparse(tensor):
				sparse_coo = value.tocoo()
				indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
										  np.expand_dims(sparse_coo.col, 1)), 1)
				value = (indices, sparse_coo.data, sparse_coo.shape)
			feed_dict[tensor] = value
		session = get_session()
		updated = session.run(self.output + [self.updates_op],
							  feed_dict=feed_dict, **self.session_kwargs)
		return updated[:len(self.outputs)]

def function(inputs, outputs, updates=None, **kwargs):
	"""Instantiate a klapa function
	"""
	if kwargs:
		for key in kwargs:
			if not (has_arg(tf.Session.run, key, True) or has_arg(Function.__init__, key, True)):
				msg = 'Invalid argument "%s" passed to K.function with Tensorflow backend' % key
				raise ValueError(msg)
	Function(inputs, outputs, updates=updates, **kwargs)

def gradients(loss, variables):
	"""Return the gradient of variables wrt loss
	"""
	return tf.gradients(loss, variables, colocate_gradients_with_ops=True)

def stop_gradient(variables):
	"""Returns variables but with zero gradient wrt other variable
	"""
	if isinstance(variables, (list, tuple)):
		return map(tf.stop_gradient, variables)
	else:
		return tf.stop_gradient(variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states, go_backwards=False, 
		mask=None, constants=None, unroll=False, input_length=None):
	"""Iterate over time dimension of a tensor
	"""
	ndim = len(inputs.get_shape())
	if ndim < 3:
		return ValueError('Input should be atleast 3D')
	axes = [1, 0] + list(range(2, ndim))
	inputs = tf.transpose(inputs, (axes))

	if mask is not None:
		if mask.dtype != tf.bool:
			mask = tf.cast(mask, tf.bool)
		if len(mask.get_shape()) == ndim - 1:
			mask = expand_dims(mask)
		mask = tf.transpose(mask, axes)

	if constants is None:
		constants = []

	global uses_learning_phase
	uses_learning_phase = False

	if unroll:
		if not inputs.get_shape()[0]:
			raise ValueError('Unrolling requires a fixed number of timestamps')
		states = initial_states
		successive_states = []
		successive_outputs = []

		input_list = tf.unstack(inputs)
		if go_backwards:
			input_list.reverse()

		if mask is not None:
			mask_list = tf.unstack(mask)
			if go_backwards:
				mask_list.reverse()

			for inp, mask_t in zip(input_list, mask_list):
				output, new_states = step_function(inp, states + constants)
				if getattr(output, '_uses_learning_phase', False):
					uses_learning_phase = True

				tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(output)[1]]))

				if not successive_outputs:
					prev_output = zeros_like(output)
				else:
					prev_output = successive_outputs[-1]

				output = tf.where(tiled_mask_t, output, prev_output)
				return_states = []
				for state, new_state in zip(states, new_states):
					tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(new_state)[1]]))
					return_states.append(tf.where(tiled_mask_t, new_state, state))

				states = return_states
				successive_outputs.append(output)
				successive_states.append(states)
			last_output = successive_outputs[-1]
			new_states = successive_states[-1]
			outpus = tf.stack(successive_outputs)
		else:
			for inp in input_list:
				output, states = step_function(inp, states + constants)
				if getattr(output, '_uses_learning_phase', False):
					uses_learning_phase = True
				successive_outputs.append(output)
				successive_states.append(states)
			last_output = successive_outputs[-1]
			new_states = successive_states[-1]
			outputs = tf.stack(successive_outputs)

	else:
		if go_backwards:
			inputs = reverse(inputs, 0)

		states = tuple(initial_states)
		time_steps = tf.shape(inputs)[0]
		outputs, _ = step_function(inputs[0], initial_states + constants)
		output_ta = tensor_array_ops.TensorArray(
			dtype=outputs.dtype,
			size=time_steps,
			tensor_array_name='output_ta')
		input_ta = tensor_array_ops.TensorArray(
			dtype=inputs.dtype,
			size=time_steps,
			tensor_array_name='input_ta')
		input_ta = input_ta.unstack(inputs)
		time = tf.constant(0, dtype='int32', name='time')

		if mask is not None:
			if not states:
				raise ValueError('No initial state provided'
								'When using masking in RNN, you should provide initial states'
								'(and your step function should return as its first state time t'
								' the output at time t-1)')
			if go_backwards:
				mask = reverse(mask, 0)

			mask_ta = tensor_array_ops.TensorArray(dtype=tf.bool, size=time_steps, tensor_array_name='mask_ta')
			mask_ta = mask_ta.unstack(mask)

			def _step(time, output_ta_t, *states):
				"""RNN step function
				"""
				current_input = input_ta.read(time)
				mask_t = mask_ta.read(time)
				output, new_states = step_function(current_input, tuple(states) + tuple(constants))

				if getattr(output, '_uses_learning_phase', False):
					global uses_learning_phase
					uses_learning_phase = True
				for state, new_state in zip(states, new_states):
					new_state.set_shape(state.get_shape())
				tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(output)[1]]))
				output = [tf.where(tiled_mask_t, new_states[i], states[i]) for i in range(len(states))]
				output_ta_t = output_ta_t.write(time, output)
				return (time + 1, output_ta_t) + tuple(new_states)
		else:
			def _step(time, output_ta_t, *states):
				"""RNN step function
				"""
				current_input = input_ta.read(time)
				output, new_states = step_function(current_input, tuple(states) + tuple(constants))

				if getattr(output, '_uses_learning_phase', False):
					global uses_learning_phase
					uses_learning_phase = True
				for state, new_state in zip(states, new_states):
					new_state.set_shape(state.get_shape())
				output_ta_t = output_ta_t.write(time, output)
				return (time + 1, output_ta_t) + tuple(new_states)

		final_outputs = control_flow_ops.while_loop(
			cond=lambda time, *_: time < time_steps,
			body = _step,
			loop_vars = (time, output_ta) + states,
			parallel_iterations = 32,
			swap_memory = True)
		last_time = final_outputs[0]
		output_ta = final_outputs[1]
		new_states = final_outputs[2:]

		outputs = output_ta.stack()
		last_output = output_ta.read(last_time -1 )

	axes = [1, 0] + list(range(2, len(outputs.get_shape())))
	outputs = tf.transpose(outputs, axes)
	last_output._uses_learning_phase = uses_learning_phase
	return last_output, outputs, new_states

def switch(condition, then_expression, else_expression):
	"""Switches between two operations depending on a scalar value
	Both then_expression and else_expression should of same shape
	"""
	if condition.dtype != tf.bool:
		condition = tf.cast(condition, 'bool')
	cond_ndim = ndim(condition)
	if not cond_ndim:
		if not callable(then_expression):
			def then_expression_fn():
				return then_expression
		else:
			then_expression_fn = then_expression
		if not callable(else_expression):
			def else_expression_fn():
				return else_expression
		else:
			else_expression_fn = else_expression
		x = tf.cond(condition, then_expression_fn, else_expression_fn)

	else:
		if callable(then_expression):
			then_expression = then_expression()
		if callable(else_expression):
			else_expression = else_expression()
		expr_ndim = ndim(then_expression)

		if cond_ndim > expr_ndim:
			raise ValueError('Rank of condition should be less than or equal to rank of then_expression'
				' and else_expression. ndim(condition) = ' + str(cond_ndim) + ', ndim(then_expression)=' + str(expr_ndim))

		if cond_ndim > 1:
			ndim_diff = expr_ndim - cond_ndim
			cond_shape = tf.concat([tf.shape(condition), [1] * ndim_diff], axis=0)
			condition = tf.reshape(condition, cond_shape)
			expr_shape = tf.shape(then_expression)
			shape_diff = expr_shape - cond_shape
			tile_shape = tf.where(shape_diff > 0, expr_shape, tf.ones_like(expr_shape))
			condition = tf.tile(condition, tile_shape)
		x = tf.where(condition, then_expression, else_expression)
	return x

def in_train_phase(x, alt, training=None):
	"""Select x in train phase, alt other wise. x and alt should have the same shape
	"""
	if training is None:
		training = learning_phase()
		uses_learning_phase = True
	else:
		uses_learning_phase = False

	if training is 1 or training is True:
		if callable(x):
			return x()
		else:
			return x

	elif training is 0 or training is False:
		if callable(alt):
			return alt()
		else:
			return alt
	x = switch(training, x, alt)
	if uses_learning_phase:
		x._uses_learning_phase = true
	return x

def in_test_phase(x, alt, training=None):
	"""Select x in test phase, alt otherwise. x and alt should be of same shape
	"""
	return in_train_phase(x, alt, training=training)

# NN Operations

def relu(x, alpha=0., max_value=None):
	"""rectified linear unit
	with default values it returns max(x, 0)
	"""
	if alpha != 0:
		negative_part = tf.nn.relu(-x)
	x = tf.nn.relu(x)
	if max_value is not None:
		max_value = _to_tensor(max_value, x.dtype.base_dtype)
		zero = _to_tensor(0., x.dtype.base_dtype)
		x = tf.clip_by_value(x, zero, max_value)
	if alpha != 0:
		alpha = _to_tensor(alpha, x.dtype.base_dtype)
		x -= alpha * negative_part
	return x

def elu(x, alpha=1.):
	"""Exponential linear unit
	"""
	res = tf.nn.elu(x)
	if alpha == 1:
		return res
	else:
		return tf.where(x > 0, res, alpha * res)

def softmax(x):
	"""Softmax on tensor
	"""
	return tf.nn.softmax(x)

def softplux(x):
	"""softplus of a tensor
	"""
	return tf.nn.softplus(x)

def softsign(x):
	"""softsign of a tensor
	"""
	return tf.nn.softsign(x)

def categorical_crossentropy(target, output, from_logits=False):
	"""Categorical cross entropy between output tensor and target tensor
	"""
	if not from_logits:
		output /= tf.reduce_sum(output, axis=len(output.get_shape()) -1, keep_dims=True)
		_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		return - tf.reduce_sum(target * tf.log(output), axis=len(output.get_shape()) - 1)
	else:
		return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)

def sparse_categorical_entropy(target, output, from_logits=False):
	"""Categorical cross entropy with integer targets
	"""
	if not from_logits:
		_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
		output = tf.log(output)

	output_shape = output.get_shape()
	targets = cast(flatten(target), 'int64')
	logits = tf.reshape(output, [-1, int(output_shape[-1])])
	res = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
	if len(output_shape) == 3:
		return tf.reshape(res, tf.shape(output)[:-1])
	else:
		return res

def binary_crossentropy(target, output, from_logits=False):
	"""binary crossentropy between an output and a target tensor
	"""
	if not from_logits:
		_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		output = tf.log(output)

	return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def sigmoid(x):
	"""Element wise sigmoid
	"""
	return tf.nn.sigmoid(x)

def hard_sigmoid(x):
	"""segment wise linear approximation of sigmoid
	"""
	x = (0.2 * x) + 0.5
	zero = _to_tensor(0., x.dtype.base_dtype)
	one = _to_tensor(1., x.dtype.base_dtype)
	x = tf.clip_by_value(x, zero, one)
	return x

def tanh(x):
	"""element wise tanh
	"""
	return tf.nn.tanh(x)

def dropout(x, level, noise_shape=None, seed=None):
	"""Set entries in x to zero at random while scaling the entire tensor
	"""
	retain_prob = 1. - level
	if seed is None:
		seed = np.random.randint(10e6)
	return tf.nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)

def l2_normalize(x, axis=None):
	"""Normalize a tensor wrt L2 norm alongside the specified axis
	"""
	return tf.nn.l2_normalize(x, dim=axis)

def in_top_k(predictions, targets, k):
	"""return whether the targets are in top k prediction
	"""
	return tf.nn.in_top_k(predictions, targets, k)

# CONVOLUTIONS

def _preprocess_deconv3d_output_shape(x, shape, data_format):
	"""Get the output_shape for the 3D deconvolution
	"""
	if data_format == 'channel_first':
		shape = (shape[0], shape[2], shape[3], shape[4], shape[1])

	if shape[0] is None:
		shape = (tf.shape(x)[0], ) + tuple(shape[1:])
		shape = tf.stack(list(shape))
	return shape

def _preprocess_deconv_output_shape(x, shape, data_format):
	"""Get the output shape of the deconvolution
	"""
	if data_format == 'channel_first':
		shape = (shape[0], shape[2], shape[3], shape[1])

	if shape[0] is None:
		shape = (tf.shape(x)[0], ) + tuple(shape[1:])
		shape = tf.stack(list(shape))
	return shape

def _preprocess_conv2d_input(x, data_format):
	"""Transpose and cast the input before the conv2d
	"""
	if dtype(x) == 'float64':
		x = tf.cast(x, 'float32')
	if data_format == 'channels_first':
		x = tf.transpose(x, (0, 2, 3, 1))
	return x

def _preprocess_conv3d_input(x, data_format):
	"""Transpose and cast the input before the conv3d
	"""
	if dtype(x) == 'float64':
		x = tf.cast('float32')

	if data_format == 'channels_first':
		x = tf.transpose(x, (0, 2, 3, 4, 1))
	return x

def _preprocess_conv2d_kernel(kernel, data_format):
	"""Transpose and cast the kernel before the conv2d
	"""
	if dtype(kernel) == 'float64':
		kernel = tf.cast(kernel, 'float32')
	if data_format == 'channel_first':
		kernel = tf.transpose(kernel, (2, 3, 1, 0))
	return kernel
	



























































