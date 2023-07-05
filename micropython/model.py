import sensor
import image
import os
import tf

# Define the paths to the positive and negative image directories
positive_images_dir = "positive_images/"
negative_images_dir = "negative_images/"
neutral_images_dir = "neutral_images/"

# Define the path to store the trained classifier
classifier_path = "hickory_stick_classifier.tflite"

# Define the dimensions of the positive and negative images for training
image_size = (48, 48)

# Create an array to store the positive, negative, and neutral samples
positive_samples = []
negative_samples = []
neutral_samples = []

# Load positive images and resize them
for filename in os.listdir(positive_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = image.Image(filename=positive_images_dir + filename)
        img.resize(image_size)
        positive_samples.append(img)

# Load negative images and resize them
for filename in os.listdir(negative_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = image.Image(filename=negative_images_dir + filename)
        img.resize(image_size)
        negative_samples.append(img)

# Load neutral images and resize them
for filename in os.listdir(neutral_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = image.Image(filename=neutral_images_dir + filename)
        img.resize(image_size)
        neutral_samples.append(img)

# Check number of training images
print("Number of positive samples:", len(positive_samples))
print("Number of negative samples:", len(negative_samples))
print("Number of neutral samples:", len(neutral_samples))

if len(positive_samples) == 0:
    print("No positive samples")
if len(negative_samples) == 0:
    print("No negative samples")
if len(neutral_samples) == 0:
    print("No neutral samples")

# Create arrays for positive, negative, and neutral labels
positive_labels = [1] * len(positive_samples)
negative_labels = [0] * len(negative_samples)
neutral_labels = [0] * len(neutral_samples)

# Concatenate positive, negative, and neutral samples and labels
samples = positive_samples + negative_samples + neutral_samples
labels = positive_labels + negative_labels + neutral_labels

# Create the neural network for training
network = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the network
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert the images to numpy arrays
samples = [img.to_grayscale().numpy() for img in samples]
samples = [img.reshape(-1) for img in samples]
samples = [img.astype('float32') for img in samples]

# Convert the labels to numpy arrays
labels = tf.convert_to_tensor(labels)

# Train the classifier
network.fit(samples, labels, epochs=10)

# Save the trained classifier
network.save(classifier_path)
print("Model saved")
