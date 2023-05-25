import numpy as np
import pandas as pd
import os
import os
import shutil
import pathlib
import random
import datetime
import cv2

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pathlib

#######################################################################
################# Data Preprocessing ##################################

""" for dirname, _, filenames in os.walk("D:/Study/Thesis"):
    for filename in filenames:
        print(os.path.join(dirname, filename))  """

classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
dir_path='D:/Study/Thesis/EyePacs dataset/eyepacs_preprocess'
os.listdir(dir_path)

df = pd.read_csv('D:/Study/Thesis/EyePacs dataset/trainLabels.csv')
df['path'] = [dir_path+'/'+img for img in os.listdir(dir_path)]
# print(len(df))
train_num = int(0.7*len(df))
train = df.iloc[:train_num,:]
test = df.iloc[train_num:,:]

""" print(len(train))
print(len(test)) """

###########################################################################
########################### MODEL BUILDING ################################

IMAGE_SIZE = [1024, 1024]

res=ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in res.layers:
    layer.trainable = False

folders=4

# our layers - you can add more if you want
x = Flatten()(res.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(4, activation='softmax')(x)

# create a model object
model = Model(inputs=res.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

###########################################
# fit the model
r = model.fit_generator(
  train,
  validation_data=test,
  epochs=5,
  steps_per_epoch=len(train),
  validation_steps=len(test)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('firstTestRes.h5')