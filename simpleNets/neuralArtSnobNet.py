
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None


board = keras.callbacks.TensorBoard(log_dir='./logs/artsnobNet', 
	histogram_freq=0, 
	batch_size=100, 
	write_graph=True)






model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
conv1 = model.add(Conv2D(32, (3, 3), input_shape=(175, 175, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

conv2 = model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))



model.compile(loss = 'binary_crossentropy', 
	optimizer = 'rmsprop', 
	metrics = ['accuracy'])
# model.summary()

datagen = ImageDataGenerator(rescale = 1./255,
	rotation_range=30, 
	width_shift_range=0.3, 
	height_shift_range = 0.3,  
	zoom_range = 0.2, 
	horizontal_flip = True, 
	fill_mode = 'nearest')



batch_size = 100 #change this from 16

train_datagen = datagen

test_datagen = ImageDataGenerator(rescale = 1./255, 
	rotation_range=30, 
	width_shift_range=0.3, 
	height_shift_range = 0.3,  
	zoom_range = 0.2, 
	horizontal_flip = True, 
	fill_mode = 'nearest')


train_generator = train_datagen.flow_from_directory('/home/brett/BigData/trainSet', 
	target_size = (175, 175), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest')


v_generator = test_datagen.flow_from_directory('/home/brett/BigData/testSet', 
	target_size = (175,175), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest')


model.fit_generator(train_generator, 
	steps_per_epoch = 2500 // batch_size, 
	epochs = 100, 
	validation_data =v_generator, 
	validation_steps = 1250 // batch_size, 
	callbacks = [board],
	use_multiprocessing=True,
    workers=8)
model.save_weights('neuralArtSnobNet.h5')

