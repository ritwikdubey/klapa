from __future__ import print_function
import klapa
from klapa.datasets import cifar10
from klapa.preprocessing.image import ImageDataGenerator
from klapa.models import Sequential
from klapa.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import pickle
import numpy as np

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'klapa_cifar10_trained_models.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = klapa.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
                optimizers=opt,
                metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                validation_data=(x_test, y_test), shuffle=True)
else:
    print('Using real time data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # Set each sample mean to 0
        featurewise_std_normalization=False, # divide input by std of the dataset
        samplewise_std_normalization=False, # divide each input by std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=0, # randomly rotate image in the range
        width_shift_range=0.1, # randomly shift image horizontally
        height_shift_range=0.1 # randomly shift image vertically
        horizontal_flip=True, # randomly flip images
        vertical_flip=False)

    # Compute quantities required for featurewise normalization
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Save trained model at %s', % model_path)

    label_list_path = 'dataset/cifar-10-batches-py/batches.meta'

    klapa_dir = os.path.expanduser(os.path.join('-', '.klapa'))
    datadir_base = os.path.expanduser(klapa_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.klapa')
    label_list_path = os.path.join(datadir_base, label_list_path)

    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)

    evaluation = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False),
                                        steps=x_test.shape[0] // batch_size,
                                        workers=4)
    print('Model accuracy = %.2f' % (evaluation[1]))

    predict_gen = model.predict_generator(datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False),
                                            steps=x_test.shape[0] // batch_size,
                                            workers=4)
    
    for predict_index, predicted_y in enumerate(predict_gen):
        actual_label = labels['label_names'][np.argmax(y_test[predict_index])]
        predicted_label = labels['label_names'][np.argmax(predicted_y)]
        print('Actual label = %s vs Predicted label = %s' % (actual_label, predicted_label))
        if predict_index == num_predictions:
            break