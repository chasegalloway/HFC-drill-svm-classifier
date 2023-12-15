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
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

pos_im_path = r"ai/training_data/positive"
neg_im_path= r"ai/training_data/negative"

pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing)
num_neg_samples = size(neg_im_listing)
print(num_pos_samples)
print(num_neg_samples)
data= []
labels = []

for file in pos_im_listing:
    img = Image.open(pos_im_path + '\\' + file)
    gray = img.convert('L')
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)
    
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    gray= img.convert('L')
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)

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

joblib.dump(model, r'ai/output/model.npy')
