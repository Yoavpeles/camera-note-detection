import cv2
import numpy as np
import constants as con

# Specify the camera index (usually 0 for built-in webcam)
CAMERA_INDEX = 1
# Define lower and upper bounds for orange color in RGB
LOWER_ORANGE_RGB_ARRAY = np.array(con.LOWER_ORANGE_RGB_VALS)  # Adjust this range as needed
UPPER_ORANGE_RGB_ARRAY = np.array(con.LOWER_ORANGE_RGB_VALS)  # Adjust this range as needed


def find_largest_orange_contour(rgb_image: np.ndarray) -> np.ndarray:
    """
    Finds the largest orange contour in an RGB image
    :param rgb_image: the image to find the contour in
    :return: the largest orange contour
    """
    # Threshold the RGB image to get only orange colors
    mask = cv2.inRange(rgb_image, LOWER_ORANGE_RGB_ARRAY, UPPER_ORANGE_RGB_ARRAY)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return max(contours, key=cv2.contourArea)

def contour_is_note(contour: np.ndarray) -> bool:
    """
    Checks if the contour is shaped like a note
    :param contour: the contour to check
    :return: True if the contour is shaped like a note
    """
    # Makes sure the contour isn't some random small spec of noise
    if cv2.contourArea(contour) < con.MINIMUM_CONTOUR_AREA:
        return False

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < 0.1:  # Adjust the threshold as needed
        return False

    # Gets the smallest convex polygon that can fit around the contour
    contour_hull = cv2.convexHull(contour)
    # Fits an ellipse to the hull, and gets its area
    ellipse = cv2.fitEllipse(contour_hull)
    best_fit_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
    # Returns True if the hull is almost as big as the ellipse
    return cv2.contourArea(contour_hull) / best_fit_ellipse_area > con.CONTOUR_DISK_THRESHOLD

def main():
    # Open the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Blurs the frame to reduce noise
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # Finds contours in the RGB frame
        contour = find_largest_orange_contour(frame_blurred)
        if contour is not None and contour_is_note(contour):
            # Get the fitted ellipse
            ellipse = cv2.fitEllipse(contour)
            # Draw the ellipse
            cv2.ellipse(frame, ellipse, (255, 0, 255), 2)
            # Calculate the convex hull
            hull = cv2.convexHull(contour)
            # Draw the convex hull
            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)

            cv2.drawContours(frame, contour, 0, (255, 255, 255), 10)

            # Draw the center for the ellipse
            center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Ellipse center
            cv2.circle(frame, center, 1, (0, 0, 255), 2)  # Draw center in red

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
