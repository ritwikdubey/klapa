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















