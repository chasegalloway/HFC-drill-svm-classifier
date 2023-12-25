import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained SVM model
model = joblib.load(r'ai/HOG/output/model.npy')

# Define HOG parameters
orientations = 9
pixels_per_cell = (16, 16)  # Adjusted for better performance
cells_per_block = (2, 2)
threshold = 0.3

def detect_objects(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to a smaller size
    resized_frame = cv2.resize(gray, (1024, 1024))  # Adjusted for better performance

    # Extract HOG features from the frame
    fd = hog(resized_frame, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)

    # Reshape the feature vector to match the input shape of the SVM model
    fd = fd.reshape(1, -1)

    # Make a prediction using the SVM model
    prediction = model.predict(fd)

    return prediction

def main():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Detect objects in the frame
        prediction = detect_objects(frame)

        # Draw bounding box if an object is detected
        if prediction == 1:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

        # Display the frame with bounding box
        cv2.imshow("Object Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
