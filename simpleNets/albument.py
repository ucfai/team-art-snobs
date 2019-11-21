from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
#import matplotlib as plt
#%inline whatever this is if you're running in a notebook. couldnt get albumentation to run in my jupyter notebook.
IMG_SIZE = 500
train_set = []
imagePath = '/home/brett/BigData/trainSet/Impressionism/'

i = 5
for img in os.listdir(imagePath):
	if i > 0:
		img_array = cv2.imread(os.path.join(imagePath, img))
		new_mat = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

		train_set.append(new_mat)
		i-=1



#from documentation
def augment_and_show(aug, image):
    image = aug(image=image)['image']
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

for img in train_set:
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	augmentation = GaussNoise()
	augment_and_show(augmentation, img)


