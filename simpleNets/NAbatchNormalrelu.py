
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate

board = keras.callbacks.TensorBoard(log_dir='./logs/normalizedModel20191104', 
	histogram_freq=0, 
	batch_size=32, 
	write_graph=True)



weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(16, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay), input_shape=(500, 500, 3))) #new
model.add(Activation('relu'))
model.add(BatchNormalization()) #new
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(32, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.4))

#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()


datagen = ImageDataGenerator(rescale = 1./255,
	rotation_range=15,
	width_shift_range=0.1, 
	height_shift_range = 0.1,  
	zoom_range = 0.2,  
	fill_mode = 'nearest',
	samplewise_std_normalization = True,
	zca_whitening = True)


train_datagen = datagen

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32 #change this from 16

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)



model.compile(loss = 'categorical_crossentropy', 
	optimizer = opt_rms, 
	metrics = ['accuracy'])
# model.summary()


train_generator = train_datagen.flow_from_directory('/home/brett/BigData/trainSet', 
	target_size = (500, 500), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest', 
	save_to_dir='/home/brett/BigData/visualTrain',
	save_format='jpeg')


v_generator = test_datagen.flow_from_directory('/home/brett/BigData/testSet', 
	target_size = (500,500), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest')

#bb: steps per epoch updated. 

numTrainImages = 6337*3
numTestImages = 704*3

model.fit_generator(train_generator, 
	steps_per_epoch = numTrainImages // batch_size, 
	epochs = 100, 
	validation_data =v_generator, 
	validation_steps = numTestImages // batch_size, 
	callbacks = [board, LearningRateScheduler(lr_schedule)],
	use_multiprocessing = True,
	workers = 14
	)
model.save_weights('normalizedModel20191104.h5')

