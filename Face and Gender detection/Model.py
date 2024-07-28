import glob
import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Activation, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, img_to_array
from sklearn.model_selection import train_test_split

# Hyperparameters
epochs = 100
learning_rate = 1e-3
batch_size = 64
data = []
labels = []

# Load and preprocess the images
image_files = [f for f in glob.glob(r"C:\\Users\\Ahmed\\Downloads\\gender_dataset_face" + "/**/*", recursive=True)]
random.shuffle(image_files)

for img in image_files:
    image = cv2.imread(img)
    if image is not None:
        image = cv2.resize(image, (96, 96))
        image = img_to_array(image)
        data.append(image)

        label = img.split(os.path.sep)[-2]
        if label == "woman":
            labels.append(1)
        else:
            labels.append(0)

data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)

# Split the data into training and testing sets
(trainx, testx, trainy, testy) = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encode the labels
trainy = to_categorical(trainy, num_classes=2)
testy = to_categorical(testy, num_classes=2)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Build the CNN model
def build(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    chan_dim = -1  # TensorFlow backend

    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build and compile the model
model = build(width=96, height=96, depth=3, classes=2)
opt = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
H = model.fit(aug.flow(trainx, trainy, batch_size=batch_size),
              validation_data=(testx, testy),
              steps_per_epoch=len(trainx) // batch_size,
              epochs=epochs, verbose=1)

# Save the model
model.save('gender_detection.keras')

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save plot to disk
plt.savefig('plot.png')
plt.show()
