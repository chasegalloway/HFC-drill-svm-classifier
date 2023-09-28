import os
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.feature import hog
from skimage.transform import resize
from xml.etree import ElementTree as ET


# Load the .vott file to extract image paths and annotations
def load_vott_file(vott_file_path):
    tree = ET.parse(vott_file_path)
    root = tree.getroot()

    image_paths = []
    annotations = []

    for image in root.findall(".//image"):
        image_path = image.find("path").text
        image_paths.append(image_path)

        image_annotations = []
        for box in image.findall(".//box"):
            label = box.find("label").text
            x_min = float(box.find("xmin").text)
            y_min = float(box.find("ymin").text)
            x_max = float(box.find("xmax").text)
            y_max = float(box.find("ymax").text)
            image_annotations.append((label, x_min, y_min, x_max, y_max))

        annotations.append(image_annotations)

    return image_paths, annotations


# Extract HOG features from an image
def extract_hog_features(image):
    resized_image = resize(image, (128, 128))  # Resize the image to a fixed size
    hog_features, _ = hog(resized_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features


# Load and preprocess the data
def load_and_preprocess_data(vott_file_path):
    image_paths, annotations = load_vott_file(vott_file_path)

    X = []  # Feature vectors
    y = []  # Labels (1 for stick, 0 for non-stick)

    for image_path, image_annotations in zip(image_paths, annotations):
        image = io.imread(image_path)

        for annotation in image_annotations:
            label, x_min, y_min, x_max, y_max = annotation
            roi = image[int(y_min):int(y_max), int(x_min):int(x_max)]  # Region of Interest
            hog_features = extract_hog_features(roi)
            X.append(hog_features)
            y.append(1 if label == 'stick' else 0)

    return np.array(X), np.array(y)


# Split the data into training and testing sets
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


# Train an SVM classifier
def train_svm_classifier(X_train, y_train):
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    return clf


# Main function
def main():
    vott_file_path = 'C:/Users/chase/DrillAIv2/ai/training_data/vott/trainng_data1.vott'  # Replace with the path to your .vott file
    X, y = load_and_preprocess_data(vott_file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training SVM classifier...")
    classifier = train_svm_classifier(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
