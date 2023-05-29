import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import tensorflow as tf
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D


# Set the paths to the image folder and CSV file
image_folder = 'D:/Study/Thesis/EyePacs_dataset/eyepacs_preprocess'
csv_file = 'D:/Study/Thesis/EyePacs_dataset/trainLabels.csv' 

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Convert the "drlvl" column to string type
df['level'] = df['level'].astype(str)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set the image size for resizing
image_size = (224, 224)

# Load and preprocess the images for train set
train_images = []
train_labels = []
for index, row in train_df.iterrows():
    image_path = os.path.join(image_folder, row['image']+".JPEG")
    try:
        image = Image.open(image_path).resize(image_size)
        image = np.array(image) #Normalize pixel values
        train_images.append(image)
        train_labels.append(row['level'])
    except (OSError, FileNotFoundError) as e:
        print(f"Error opening image: {image_path}")
        continue

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Load and preprocess the images for test set
test_images = []
test_labels = []
for index, row in test_df.iterrows():
    image_path = os.path.join(image_folder, row['image']+".JPEG")
    try:
        image = Image.open(image_path).resize(image_size)
        image = np.array(image)
        test_images.append(image)
        test_labels.append(row['level'])
    except (OSError, FileNotFoundError) as e:
        print(f"Error opening image: {image_path}")
        continue

test_images = np.array(test_images)
test_labels = np.array(test_labels)
# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained ResNet50
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation='softmax'))  # Assuming there are 5 classes (0 to 4)

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# history = model.fit(train_images, train_labels, validation_split=0.2, epochs=10)
history = model.fit(train_images, train_labels, validation_split=0.2, epochs=10, batch_size=16)

# Evaluate the model on the test set
_, accuracy = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save('resnet_model.h5')
