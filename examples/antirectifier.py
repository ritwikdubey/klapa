from __future__ import print_function
import klapa
from klapa.models import Sequential
from klapa import layers
from klapa.datasets import mnist
from klapa import backend as K

class Antirectifier(layers.Layer):
	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 2
		shape[-1] *= 2
		return tuple(shape)

	def call(self, inputs):
		inputs -= K.mean(inputs, axis=1, keepdims=True)
		inputs = K.l2_normalize(inputs, axis=1)
		pos = K.relu(inputs)
		neg = K.relu(-inputs)
		return K.concatenate([pos, neg], axis=1)

batch_size = 128
num_classes = 10
epochs = 40

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))