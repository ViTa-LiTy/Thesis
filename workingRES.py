import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

#Next, we'll define some constants and paths based on your data structure.

data_folder = 'D:/Study/Thesis/EyePacs dataset'  # Replace with the path to your data folder
images_folder = os.path.join(data_folder, 'eyepacs_preprocess')
labels_file = os.path.join(data_folder, 'trainLabels.csv')

#Now, we'll load the labels from the CSV file and split them into training and testing sets.

labels_df = pd.read_csv(labels_file)
image_filenames = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
labels = labels_df['level'].values
# Splitting the data into training and testing sets (80% train, 20% test)
train_files, test_files, train_labels, test_labels = train_test_split(image_filenames, labels, test_size=0.2, random_state=42)

#Now, let's define a function to preprocess the images and create a data generator using Keras' ImageDataGenerator class.

def preprocess_image(image_array):
    img = tf.keras.preprocessing.image.array_to_img(image_array)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array
# Create image data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_image)

#Next, we'll create data loaders for the training and testing sets using the flow_from_directory function of the data generator.

batch_size = 32

train_data_loader = data_generator.flow_from_directory(
    images_folder,
    classes=[''],  # We use an empty string as the class since we'll provide labels separately
    batch_size=batch_size,
    target_size=(224, 224),
    shuffle=True,
    subset='training'
)
test_data_loader = data_generator.flow_from_directory(
    images_folder,
    classes=[''],
    batch_size=batch_size,
    target_size=(224, 224),
    shuffle=False,
    subset='validation'
)

#Now, let's load the pre-trained ResNet50 model without the top classification layer and add our own layers for classification.

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Assuming you have 5 classes (0-4)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit the model

""" model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10
) """

epochs = 10

model.fit(train_data_loader, epochs=epochs)

# Evaluate the model on the testing data
epochs = 10

model.fit(train_data_loader, epochs=epochs, validation_data=test_data_loader)

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(test_data_loader)

