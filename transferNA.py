
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import SGD, Adam


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None


batch_size = 20 #change this from 16


HEIGHT = 300
WIDTH = 300

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))



board = keras.callbacks.TensorBoard(log_dir='./logs/normalizedModel3', 
	histogram_freq=0, 
	batch_size=50, 
	write_graph=True)


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)
    predictions = Dense(3, activation='softmax')(x)
    finetune_model = Model(input=base_model.input, output=predictions)
    return finetune_model

FC_LAYERS = [256, 256]
dropout = .5

finetune_model = build_finetune_model(base_model,
	dropout = dropout,
	fc_layers = FC_LAYERS,
	num_classes = 3)




datagen = ImageDataGenerator(preprocessing_function= preprocess_input, 
	rescale = 1./255,
	rotation_range=40, 
	width_shift_range=0.2, 
	height_shift_range = 0.2, 
	shear_range = 0.2, 
	zoom_range = 0.2, 
	horizontal_flip = True, 
	fill_mode = 'nearest')


train_datagen = datagen

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/home/brett/BigData/trainSet', 
	target_size = (300, 300), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest')


v_generator = test_datagen.flow_from_directory('/home/brett/BigData/testSet', 
	target_size = (300, 300), 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	color_mode='rgb',
	interpolation = 'nearest')



filepath="/home/brett/artsnobs/checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint, board]

adam = Adam(lr=0.00001)

finetune_model.compile(loss = 'categorical_crossentropy', 
	optimizer = adam, 
	metrics = ['accuracy'])
# model.summary()



finetune_model.fit_generator(train_generator, 
	steps_per_epoch = 2500 // batch_size, 
	epochs = 50, 
	validation_data =v_generator, 
	validation_steps = 1250 // batch_size, 
	callbacks = callbacks_list,
	use_multiprocessing = True,
	workers = 6
	)
model.save_weights('normalizedModel3.h5')

