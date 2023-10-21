import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# SVM implementation using NumPy
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.num_epochs):
            misclassified_count = 0

            for i in range(n_samples):
                # Update rule for misclassified samples
                if y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]
                    misclassified_count += 1

            print(f"Epoch {epoch + 1}, Misclassified samples: {misclassified_count}")

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)

# Define the paths to the positive, negative, and neutral image directories
positive_images_dir = r"ai/training_data/positive"
negative_images_dir = r"ai/training_data/negative"
neutral_images_dir = r"ai/training_data/neutral"

image_size = (500, 500)

# Load and process images with augmentation
def load_and_process_images(image_dir):
    samples = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image = cv2.imread(os.path.join(image_dir, filename))
            image = cv2.resize(image, image_size)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            samples.append(gray)

            # Data Augmentation
            # Rotate by 90 degrees
            rotated_90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            samples.append(rotated_90)

            # Rotate by 180 degrees
            rotated_180 = cv2.rotate(gray, cv2.ROTATE_180)
            samples.append(rotated_180)

            # Rotate by 270 degrees
            rotated_270 = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            samples.append(rotated_270)

            # Flip horizontally
            flipped_horizontal = cv2.flip(gray, 1)
            samples.append(flipped_horizontal)

            # Flip vertically
            flipped_vertical = cv2.flip(gray, 0)
            samples.append(flipped_vertical)

            # Add noise
            noisy_image = gray + np.random.normal(0, 25, gray.shape).astype(np.uint8)
            samples.append(noisy_image)

    return samples

positive_samples = load_and_process_images(positive_images_dir)
negative_samples = load_and_process_images(negative_images_dir)
neutral_samples = load_and_process_images(neutral_images_dir)

# Convert the samples to NumPy arrays
positive_samples = np.array(positive_samples)
negative_samples = np.array(negative_samples)
neutral_samples = np.array(neutral_samples)

# Flatten the samples and create labels
positive_labels = np.ones(len(positive_samples))
negative_labels = np.zeros(len(negative_samples))
neutral_labels = np.zeros(len(neutral_samples))

# Concatenate samples and labels
samples = np.vstack((positive_samples, negative_samples, neutral_samples))
labels = np.hstack((positive_labels, negative_labels, neutral_labels))

# Reshape samples
num_samples, _, _ = samples.shape
samples = samples.reshape(num_samples, -1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train the model
svm = SVM()
svm.fit(X_train, y_train)

# Predict on the validation set
val_predictions = svm.predict(X_val)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy * 100, "%")

# Save the trained classifier using XML file
classifier_path = "ai/training_data/newmodel.xml"

# Create an SVM object and set the trained parameters
svm_classifier = cv2.ml.SVM_create()
svm_classifier.setKernel(cv2.ml.SVM_LINEAR)
svm_classifier.setType(cv2.ml.SVM_C_SVC)
svm_classifier.setC(1.0)

# Set the support vectors and other necessary parameters
svm_classifier.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

# Adjust class weights for binary classification
class_weights = np.array([1.0, 1.0])
svm_classifier.setClassWeights(class_weights)

# Convert X_train to CV_32F data type
X_train = X_train.astype(np.float32)

# Convert y_train to integer type (CV_32S)
y_train = y_train.astype(np.int32)

# Train the SVM model with the training data
svm_classifier.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Save the trained SVM model to an XML file
svm_classifier.save(classifier_path)

print("Model saved to", classifier_path)
