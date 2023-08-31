import cv2

# Define the path to the trained classifier
classifier_path = "training_data/classifier.xml"

# Load the trained classifier for detection
svm = cv2.ml.SVM_load(classifier_path)

# Create the Haar cascade classifier for detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'stick', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 2)

    # Display the processed frame on a screen
    cv2.imshow('Frame', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Key actions
    if key == ord('q'):
        running = False
    if key == ord('w'):
        print("w")
    if key == ord('e'):
        print("e")
    if key == ord('r'):
        print("r")

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
