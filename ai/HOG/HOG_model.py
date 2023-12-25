from skimage.feature import hog
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
from numpy import *

orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
threshold = .3

pos_im_path = r"ai/training_data/positive"
neg_im_path = r"ai/training_data/negative"

pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing)
num_neg_samples = size(neg_im_listing)
print("Original Positive Samples:", num_pos_samples)
print("Original Negative Samples:", num_neg_samples)

data = []
labels = []
image_size = (1024, 1024)

# Load and process positive samples
for file in pos_im_listing:
    img = Image.open(pos_im_path + '\\' + file)
    gray = img.convert('L')

    # Resize the image
    gray_resized = gray.resize(image_size, Image.LANCZOS)

    fd = hog(gray_resized, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    # Data Augmentation
    rotated_90 = gray_resized.rotate(90)
    fd = hog(rotated_90, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    rotated_180 = gray_resized.rotate(180)
    fd = hog(rotated_180, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    rotated_270 = gray_resized.rotate(270)
    fd = hog(rotated_270, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    flipped_horizontal = gray_resized.transpose(Image.FLIP_LEFT_RIGHT)
    fd = hog(flipped_horizontal, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    flipped_vertical = gray_resized.transpose(Image.FLIP_TOP_BOTTOM)
    fd = hog(flipped_vertical, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

    noisy_image = np.array(gray_resized) + np.random.normal(0, 25, np.array(gray_resized).shape).astype(np.uint8)
    fd = hog(noisy_image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

# Load and process negative samples
for file in neg_im_listing:
    img = Image.open(neg_im_path + '\\' + file)
    gray = img.convert('L')

    # Resize the image
    gray_resized = gray.resize(image_size, Image.LANCZOS)

    fd = hog(gray_resized, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    # Data Augmentation
    rotated_90 = gray_resized.rotate(90)
    fd = hog(rotated_90, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    rotated_180 = gray_resized.rotate(180)
    fd = hog(rotated_180, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    rotated_270 = gray_resized.rotate(270)
    fd = hog(rotated_270, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    flipped_horizontal = gray_resized.transpose(Image.FLIP_LEFT_RIGHT)
    fd = hog(flipped_horizontal, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    flipped_vertical = gray_resized.transpose(Image.FLIP_TOP_BOTTOM)
    fd = hog(flipped_vertical, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

    noisy_image = np.array(gray_resized) + np.random.normal(0, 25, np.array(gray_resized).shape).astype(np.uint8)
    fd = hog(noisy_image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)

num_pos_samples_after_aug = num_pos_samples * 7
num_neg_samples_after_aug = num_neg_samples * 7  

print("Positive Samples after Augmentation:", num_pos_samples_after_aug)
print("Negative Samples after Augmentation:", num_neg_samples_after_aug)

le = LabelEncoder()
labels = le.fit_transform(labels)

print(" Constructing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), labels, test_size=0.20, random_state=42)

print(" Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

joblib.dump(model, r'ai/HOG/output/model.npy')
