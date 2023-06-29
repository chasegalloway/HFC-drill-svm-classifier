import cv2
import numpy as np
import os

# Define the paths to the positive and negative image directories
positive_images_dir = "Training Data/positive_images/"
negative_images_dir = "Training Data/negative_images/"
neutral_images_dir = "Training Data/neutral_images/"

# Define the path to store the trained classifier
classifier_path = "Training Data/hickory_stick_classifier.xml"

# Define the dimensions of the positive and negative images for training
image_size = (4032, 3024)

# Create an array to store the positive, negative, and neutral samples
positive_samples = []
negative_samples = []
neutral_samples = []

# Load positive images and resize them
for filename in os.listdir(positive_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(os.path.join(positive_images_dir, filename))
        image = cv2.resize(image, image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        positive_samples.append(gray)

# Load negative images and resize them
for filename in os.listdir(negative_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(os.path.join(negative_images_dir, filename))
        image = cv2.resize(image, image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        negative_samples.append(gray)

# Load neutral images and resize them
for filename in os.listdir(neutral_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(os.path.join(neutral_images_dir, filename))
        image = cv2.resize(image, image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        neutral_samples.append(gray)

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
positive_labels = np.ones(len(positive_samples), dtype=int)
negative_labels = np.zeros(len(negative_samples), dtype=int)
neutral_labels = np.zeros(len(neutral_samples), dtype=int)

# Concatenate positive, negative, and neutral samples and labels
samples = np.concatenate((positive_samples, negative_samples, neutral_samples))
labels = np.concatenate((positive_labels, negative_labels, neutral_labels))

# Create the Haar cascade classifier for detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the SVM classifier for training
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Reshape the samples array
num_samples, _, _ = samples.shape
samples = samples.reshape(num_samples, -1)

# Train the classifier
samples = samples.astype(np.float32)  # Convert samples to CV_32F
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

# Save the trained classifier
svm.save(classifier_path)
print("Model saved")
