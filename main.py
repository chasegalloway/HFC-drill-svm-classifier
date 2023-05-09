import cv2
import numpy as np
import os

# Define the paths to the positive and negative image directories
positive_images_dir = "positive_images/"
negative_images_dir = "negative_images/"
neutral_images_dir = "neutral_images/"

# Define the path to store the trained classifier
classifier_path = "hickory_stick_classifier.xml"

# Define the dimensions of the positive and negative images for training
image_size = (50, 50)

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

# Train the classifier
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

# Save the trained classifier
svm.save(classifier_path)

# Load the trained classifier for detection
svm = cv2.ml.SVM_load(classifier_path)

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video stream
cap.set(4, 480)  # Set the height of the video stream

# Flag variable to control the loop
running = True

# Loop to continuously capture frames from the camera and detect hickory sticks
while running:
    # Read the frame from the video capture
    ret, frame = cap.read()

# Convert the frame to grayscale for detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect hickory sticks in the frame
sticks = detector.detectMultiScale(gray)

# Draw bounding boxes around the detected hickory sticks
for (x, y, w, h) in sticks:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 75, 0), 2)

# Display the processed frame on a screen
cv2.imshow('Frame', frame)

# Check for key press
key = cv2.waitKey(1) & 0xFF

# If key 'q' is pressed, set the running flag to False
if key == ord('q'):
    running = False

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()