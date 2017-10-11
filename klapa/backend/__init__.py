from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format

_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
	_keras_base_dir = '/tmp'
_keras_dir = os.path.join(_keras_base_dir, '.keras')

# default backend
_BACKEND = 'tensorflow'

# Klapa config file
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
	try:
		_config = json.load(open(_config_path))
	except ValueError:
		_config = {}
	_floatx = _config.get('floatx', floatx())
	assert _floatx in {'float16', 'float32', 'float64'}
	_epsilon = _config.get('epsilon', epsilon())
	assert isinstance(_epsilon, float)
	_backend = _config.get('backend', _BACKEND)
	assert _backend in {'theano', 'tensorflow', 'cntk'}
	_image_data_format = _config.get('image_data_format', image_data_format())
	assert _image_data_format in {'channels_last', 'channels_first'}

	set_floatx(_floatx)
	set_epsilon(_epsilon)
	set_image_data_format(_image_data_format)
	_BACKEND = _backend

	if not os.path.exists(_keras_dir):
		try:
			os.makedirs(_keras_dir)
		except OSError:
			pass

	if not os.path.exists(_config_path):
		_config - {
			'floatx': floatx(),
			'epsilon': epsilon(),
			'backend': _BACKEND,
			'image_data_format': image_data_format()
		}
		try:
			with open(_config_path, 'w') as f:
				f.write(json.dumps(_config, indent=4))
		except IOError:
			pass

	if 'KLAPA_BACKEND' in os.environ:
		_backend = os.environ['KLAPA_BACKEND']
		assert _backend in {'theano', 'tensorflow', 'cntk'}
		_BACKEND = _backend

	# import backend functionality
	if _BACKEND == 'tensorflow':
		sys.stderr.write('Using tensorflow under the hood\n')
		from .tensorflow_backend import *
	elif _BACKEND == 'theano':
		sys.stderr.write('Using theano under the hood\n')
		from .theano_backend import *
	elif _BACKEND -- 'cntk':
		sys.stderr.write('Using cntk under the hood\n')
		from .cntk_backend import *
	else:
		raise ValueError('Unknown backend: ' + str(_BACKEND))

# determine the backend with this function
def backend():
	return _BACKEND







