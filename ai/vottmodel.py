import cv2
import numpy as np
import os
import json

# Define the paths to the .vott file and image directory
vott_file_path = "training_data/vott/training_data.vott"
image_dir = "training_data/images/"

# Define the path to store the trained classifier
classifier_path = "training_data/vott_classifier.xml"

# Define the dimensions of the training images
image_size = (4032, 3024)

# Create arrays to store samples and labels
samples = []
labels = []

# Load the .vott file and parse it
with open(vott_file_path, "r") as vott_file:
    vott_data = json.load(vott_file)

# Iterate through the vott data to extract samples and labels
for item in vott_data["assets"]:
    image_filename = item["asset"]["name"]
    image_path = os.path.join(image_dir, image_filename)

    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        samples.append(gray)

        # Determine the label based on the tag name in the .vott file
        tag_name = item["regions"][0]["tags"][0]
        if tag_name == "positive":
            labels.append(1)
        elif tag_name == "negative":
            labels.append(0)
        elif tag_name == "neutral":
            labels.append(0)  # You may want to change this label based on your specific use case

# Check the number of training samples
print("Number of samples:", len(samples))

if len(samples) == 0:
    print("No samples found in the .vott file")

# Create the Haar cascade classifier for detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the SVM classifier for training
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Reshape the samples array
num_samples, _, _ = np.array(samples).shape
samples = np.array(samples).reshape(num_samples, -1)

# Train the classifier
samples = samples.astype(np.float32)  # Convert samples to CV_32F
labels = np.array(labels)
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

# Save the trained classifier
svm.save(classifier_path)
print("Model saved")
