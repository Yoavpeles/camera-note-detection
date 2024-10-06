import cv2
import numpy as np

# This script is to help find the lower and upper RGB bounds for object detection.
# Green represents a contour that has been found, blue represents the biggest contour. 
# Adjust the trackbars to only highlight the objects of interest.

# Create a window to adjust the lower and upper bounds
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing
cv2.resizeWindow("Trackbars", 600, 300)  # Set the size of the window (width, height)

# Create trackbars for RGB lower and upper bounds
cv2.createTrackbar("Red Lower", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("Green Lower", "Trackbars", 100, 255, lambda x: None)
cv2.createTrackbar("Blue Lower", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("Red Upper", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("Green Upper", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("Blue Upper", "Trackbars", 255, 255, lambda x: None)

# Specify the camera index (usually 0 for built-in webcam)
camera_index = 1

# Open the camera
cap = cv2.VideoCapture(camera_index)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Get current trackbar positions
        red_lower = cv2.getTrackbarPos("Red Lower", "Trackbars")
        green_lower = cv2.getTrackbarPos("Green Lower", "Trackbars")
        blue_lower = cv2.getTrackbarPos("Blue Lower", "Trackbars")
        red_upper = cv2.getTrackbarPos("Red Upper", "Trackbars")
        green_upper = cv2.getTrackbarPos("Green Upper", "Trackbars")
        blue_upper = cv2.getTrackbarPos("Blue Upper", "Trackbars")

        # Define lower and upper bounds for orange color in RGB
        lower_orange = np.array([blue_lower, green_lower, red_lower])
        upper_orange = np.array([blue_upper, green_upper, red_upper])

        # Threshold the RGB image to get only the specified colors
        mask = cv2.inRange(frame, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (clump) of specified color pixels
        if contours:
            # Draws all detected contours in green
            cv2.drawContours(frame, contours, -1, [0, 255, 0], 1)
            # Gets the largest contour and draws it in blue
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], 0, [255, 0, 0], 2)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Break the loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Unable to capture frame")
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
