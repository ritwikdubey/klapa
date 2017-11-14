import six
import warnings
import functools
import numpy as np

def general_legacy_interface(allowed_position_args=None, conversation=None,
							conversations=None, preprocessor=None, value_conversation=None,
							object_type='class'):
	if allowed_position_args is None:
		check_positional_args = False
	else:
		check_positional_args = True

	allowed_position_args = allowed_position_args or []
	conversations conversations or []
	value_conversation = value_conversation or []

	def legacy_support(func):
		@six.wraps(func):
		def wrapper(*args, **kwargs):
			if object_type == 'class':
				object_name = args[0].__class__.__name__
			else:
				object_name = func.__name__
			if preprocessor:
				args, kwargs, converted = preprocessor(args, kwargs)
			else:
				converted = []
			if check_positional_args:
				if len(args) > len(allowed_position_args) + 1:
					raise TypeError('`' + object_name + '` can only accept ' + 
									str(len(allowed_position_args)) + ' positional arguments ' + 
									str(tuple(allowed_position_args)) + ' but you passed the following ' + 
									'positional arguments: ' + str(list(args[1:])))
			for key in value_conversation:
				if key in kwargs:
					old_value = kwargs[key]
					if old_value in value_conversation[key]:
						kwargs[key] = value_conversation[key][old_value]
			for old_name, new_name in conversations:
				if old_name in kwargs:
					value = kwargs.pop(old_name)
					if new_name in kwargs:
						raise_duplicate_arg_error(old_name, new_name)
					kwargs[new_name] = value
					converted.append((new_name, old_name))

			if converted:
				signature = '`' + object_name + '('
				for i, value in enumerate(args[1:]):
					if isinstance(value, six.string_types):
						signature += '"' + value + '"'
					else:
						if isinstance(value, np.ndarray):
							str_val = 'array'
						else:
							str_val = str(value)
						if len(str_val) > 10:
							str_val = str_val[:10] + '...'
						signature += str_val
					if i < len(args[1:]) - 1 or kwargs:
						signature += ', '
				for i, (name, value) in enumerate(kwargs.items()):
					signature += name + '='
					if isinstance(value, six.string_types):
						signature +='"' + value '"'
					else:
						if isinstance(value, np.ndarray):
							str_val = 'array'
						else:
							str_val = str(value)
						if len(str_val) > 10:
							str_val = str_val[:10] + '...'
						signature += str_val
					if i < len(kwargs) - 1:
						signature += ', '
				signature += ')`'
				warnings.warn('Update your `' + object_name +
							'`call to the keras 2 API:' + signature, stacklevel=2)
			return func(*args, **kwargs)
		wrapper._original_function = func
		return wrapper
	return legacy_support

generate_legacy_method_interface = functools.partial(generate_legacy_interface, object_type='method')

def raise_duplicate_arg_error(old_arg, new_arg):
	raise TypeError('For the ' + new_arg + ' argument, the layer received both '
					'the legacy keyword argument ' + old_arg + ' and the keras 2 keyword argument '
					+ new_arg)

legacy_dense_support = generate_legacy_interface(
	allowed_position_args=['units'],
	conversions=[('output_dim', 'units'),
				('init', 'kernel_initializer'),
				('W_regularizer', 'kernel_regularizer'),
				('b_regularizer', 'bias_regularizer'),
				('W_constraint', 'kernel_constraint'),
				('b_constraint', 'bias_constraint'),
				('bias', 'use_bias')])

legacy_dropout_support = generate_legacy_interface(
	allowed_position_args=['rate', 'noise_shape', 'seed'],
	conversions=[('p', 'rate')])

def recurrent_args_preprocessor(args, kwargs):
	converted = []
	if 'forget_bias_init' in kwargs:
		if kwargs['forget_bias_init'] == 'one':
			kwargs.pop('forget_bias_init')
			kwargs['unit_forget_bias'] = True
			converted.append(('forget_bias_init', 'unit_forget_bias'))
		else:
			kwargs.pop('forget_bias_init')
			warnings.warn('The forget_bias_unit has been ignored', stacklevel=3)

	if 'input_dim' in kwargs:
		input_length = kwargs.pop('input_length', None)
		input_dim = kwargs.pop('input_dim')
		input_shape = (input_length, input_dim)
		kwargs['input_shape'] = input_shape
		converted.append(('input_dim', 'input_shape'))
		warnings.warn('The input_dim and input_length arguments in recurrent layers'
					'are deprecated', stacklevel=3)
	return args, kwargs, converted

legacy_recurrent_support = generate_legacy_interface(
	allowed_position_args = ['units'],
	conversions=[('output_dim', 'units'),
				('init', 'kernel_initializer'),
				('inner_init', 'recurrent_initializer'),
				('inner_activation', 'recurrent_activation'),
				('W_regularizer', 'kernel_regularizer'),
				('b_regularizer', 'bias_regularizer'),
				('U_regularizer', 'recurrent_regularizer'),
				('dropout_W', 'dropout'),
				('dropout_U', 'recurrent_dropout'),
				('consume_less', 'implementation')],
	value_conversations={'consume_less': {'cpu': 0,
										'mem': 1,
										'gpu': 2}},
	preprocessor=recurrent_args_preprocessor)





















