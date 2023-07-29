import cv2
import os
cv2.setUseOpenVX(False)
cv2.setUseOptimized(False)
# Create a VideoCapture object with the correct camera index (e.g., 0 or 1)
# Use 0 for the default camera (HDMI input) or 1 for an external camera
cap = cv2.VideoCapture(-1)

# Check if the camera is opened correctly
if not cap.isOpened():
    raise IOError("Error: Cannot open the camera.")

# Create a folder to save the images
output_folder = 'captured_images'
os.makedirs(output_folder, exist_ok=True)

# Counter to keep track of captured images
image_counter = 1
max_images = 100

try:
    while image_counter <= max_images:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            break

        # Save the frame as an image in the output folder
        image_filename = os.path.join(output_folder, f'frame_{image_counter:03d}.jpg')
        cv2.imwrite(image_filename, frame)

        print(f"Captured image {image_counter}/{max_images}")

        image_counter += 1

except KeyboardInterrupt:
    pass

# Release the capture
cap.release()
cv2.destroyAllWindows()
