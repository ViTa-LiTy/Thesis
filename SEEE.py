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
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

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
image_size = (128, 128)

# Define data augmentation parameters
data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the images for the train set with data augmentation
train_images = []
train_labels = []
for index, row in train_df.iterrows():
    image_path = os.path.join(image_folder, row['image'] + ".JPEG")
    try:
        image = Image.open(image_path).resize(image_size)
        image = np.array(image)  # Normalize pixel values
        train_images.append(image)
        train_labels.append(row['level'])
    except (OSError, FileNotFoundError) as e:
        print(f"Error opening image: {image_path}")
        continue

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Apply data augmentation only for "level" 1-4 images
level_mask = np.isin(train_labels, ['1', '2', '3', '4'])
augmented_images = []
for image in train_images[level_mask]:
    augmented = data_generator.random_transform(image)
    augmented_images.append(augmented)

augmented_images = np.array(augmented_images)
augmented_labels = train_labels[level_mask]

# Combine augmented images with original non-augmented images
train_images_augmented = np.concatenate([train_images[~level_mask], augmented_images])
train_labels_augmented = np.concatenate([train_labels[~level_mask], augmented_labels])

# Load and preprocess the images for the test set
test_images = []
test_labels = []
for index, row in test_df.iterrows():
    image_path = os.path.join(image_folder, row['image'] + ".JPEG")
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

# Convert the labels to categorical format
num_classes = 5  # Number of classes (levels)
train_labels_augmented = to_categorical(train_labels_augmented, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers[-20:]:
    layer.trainable = True


# Create a new model on top of the pre-trained EfficientNetB0
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))  # Additional dense layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_images_augmented, train_labels_augmented,
                    validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set
_, accuracy = model.evaluate(test_images, test_labels)

# Save the model
#model.save('efficientnet_model.h5')
model.save_weights('efficientnetNEWLAY20LAY_weights.h5')
