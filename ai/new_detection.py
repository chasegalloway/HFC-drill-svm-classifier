import cv2
import numpy as np

# Define the path to the trained classifier
classifier_path = r"ai/training_data/newmodel.xml"

# Load the trained classifier for detection
svm_classifier = cv2.ml.SVM_create()
svm_classifier = svm_classifier.load(classifier_path)

if svm_classifier.empty():
    print("Failed to load the classifier.")
else:
    print("Classifier Loaded")

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video stream
cap.set(4, 480)  # Set the height of the video stream

# Flag variable to control the loop
running = True

# Loop to continuously capture frames from the camera and detect objects
while running:
    # Read the frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prepare the frame for detection (resizing, data type conversion)
    gray = cv2.resize(gray, (500, 500))
    gray = gray.astype(np.float32)

    # Predict using the trained SVM model
    result = svm_classifier.predict(np.array([gray.reshape(-1)]))

    if result[1] == 1.0:
        # Object detected
        x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]  # Adjust these values as needed
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Stick', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame on a screen
    cv2.imshow('frame', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Key actions (Will be better applied later)
    if key == ord('1'):
        print("Shut down")
        running = False
    if key == ord('2'):
        print("Pause")
    if key == ord('3'):
        print("Resume")
    if key == ord('4'):
        print("Restart")
    if key == ord('5'):
        print("TBD")

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
