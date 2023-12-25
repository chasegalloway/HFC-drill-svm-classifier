import cv2
import numpy as np

def detect_objects(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for dull gray to brown
    lower_color = np.array([11, 35, 34])  # Example lower limit for dull gray
    upper_color = np.array([28, 58, 21])  # Example upper limit for brown

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

def main():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Detect objects in the frame
        result_frame = detect_objects(frame)

        # Display the original and result frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Result Frame', result_frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
