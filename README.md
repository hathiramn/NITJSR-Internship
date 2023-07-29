# NIT-Intern-PYNQ-Code
#http://192.168.2.99:9090/

## Step1: Clone the git repo

* cmd: git clone https://github.com/Xilinx/BNN-PYNQ
* Then please follow up on the document I've provided.
* The examples are provided in the testNotebook.ipynb



## Object detection 

* git clone https://github.com/Xilinx/QNN-MO-PYNQ.git
* cd QNN-MO-PYNQ
* python3 setup.py install
* Then navigate to the notebook folder for object detection


```python



import cv2
cv2.setUseOpenVX(False)
cv2.setUseOptimized(False)
# Create a VideoCapture object with the correct camera index (e.g., 0 or 1)
# Use 0 for the default camera (HDMI input) or 1 for an external camera
cap = cv2.VideoCapture(-1)

# Check if the camera is opened correctly
if not cap.isOpened():
    raise IOError("Error: Cannot open the camera.")
from IPython.display import display, clear_output,Image

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            break

        # Display the frame in the Jupyter Notebook
        clear_output(wait=True)
        _, img = cv2.imencode('.jpeg', frame)
        display(Image(data=img.tobytes()))

except KeyboardInterrupt:
    pass

# Release the capture
cap.release()
cv2.destroyAllWindows()



```
```python

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

```


```python
# live stream data saver into a folder
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


```

```python
import cv2
import os

# Create a VideoCapture object with the correct camera index (e.g., 0 or 1)
# Use 0 for the default camera (HDMI input) or 1 for an external camera
cap = cv2.VideoCapture(-1)

# Check if the camera is opened correctly
if not cap.isOpened():
    raise IOError("Error: Cannot open the camera.")

# Create a folder to save the images
output_folder = 'captured_images'
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

        # Convert the frame to grayscale (Haar cascades work on grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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


```



```python
# facetracking

import cv2
import os

# Create a VideoCapture object with the correct camera index (e.g., 0 or 1)
# Use 0 for the default camera (HDMI input) or 1 for an external camera
cap = cv2.VideoCapture(-1)

# Check if the camera is opened correctly
if not cap.isOpened():
    raise IOError("Error: Cannot open the camera.")

# Create a folder to save the images
output_folder = 'captured_images'
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

        # Convert the frame to grayscale (Haar cascades work on grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
```
