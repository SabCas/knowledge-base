# Person detection

## Camera Calibration

1. **Detecting People with YOLO:** First, you run the YOLO model on the images coming from the drone's camera. YOLO will detect objects (like people) in each frame. But without adjustments, the detection might not be perfect, especially if the camera is angled or the drone is flying high.
2. **Camera Calibration:**
Why? The drone’s camera could have distortion due to its angle or height.
How? You can calibrate the camera to understand its position relative to the ground. This is done using camera parameters like focal length and lens distortion. OpenCV, a popular Python library, can help you with this.
Use functions like cv2.calibrateCamera() to correct the distortion.
This step ensures the image isn't warped and looks more like a real-world view.
3. **Perspective Transformation (Fixing the Angle):**
Why? If the camera is not pointing straight down, objects (people) might look stretched or skewed.
How? You can apply a perspective transformation. This means transforming the image to look like you are seeing it from directly above. Again, OpenCV can help:
Use cv2.getPerspectiveTransform() and cv2.warpPerspective() to fix the angle.
These functions take the points from the camera view and map them to a flat ground view.
4. **Height Adjustment:**
Why? As the drone flies higher, people look smaller in the camera. YOLO might have trouble detecting small objects.
How? You can dynamically adjust the detection thresholds based on the drone’s altitude.
For example, if the drone is high up, you can tell YOLO to be more sensitive to smaller objects.
If you know the drone's height (from sensors or GPS), you can adjust the size of bounding boxes to better capture small objects like people from a higher perspective.
5. **Combining it All:**
The basic flow would look like this:
- Capture the image from the drone.
- Apply the camera calibration to remove distortion.
If the camera is angled, use perspective transformation to adjust the view.
- Run YOLO to detect people.
If needed, adjust the YOLO detection parameters based on drone height to ensure it detects smaller people when flying high.

``` python
import cv2
import numpy as np
import torch  # for YOLOv5

# Load YOLO model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Capture an image from the drone (example)
image = cv2.imread('drone_image.jpg')

# Step 1: Apply camera calibration to correct distortion
# (Assuming you have calibration data from a previous step)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Focal lengths and center
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Distortion coefficients
image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

# Step 2: Apply perspective transformation if the camera is at an angle
src_points = np.float32([[src1_x, src1_y], [src2_x, src2_y], [src3_x, src3_y], [src4_x, src4_y]])
dst_points = np.float32([[dst1_x, dst1_y], [dst2_x, dst2_y], [dst3_x, dst3_y], [dst4_x, dst4_y]])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
image_transformed = cv2.warpPerspective(image_undistorted, matrix, (image.shape[1], image.shape[0]))

# Step 3: Run YOLO detection on the corrected image
results = model(image_transformed)

# Step 4: Visualize detections (people should now be easier to detect)
results.show()

# Step 5: Adjust detection sensitivity if needed based on the drone's altitude
# (Adjust model thresholds here if necessary)

```


## Using Certainty Factors with YOLO on a Drone

When using a drone to detect people with YOLO, you can enhance detection accuracy by applying **certainty factors** like a Gaussian distribution. This helps account for uncertainty due to factors like distance, altitude, and camera angle.

## Why Use Certainty Factors?

1. **Varying Detection Confidence**: The drone's altitude or camera angle can affect YOLO's confidence in detecting people.
2. **Noise Handling**: Gaussian factors can help model noise (e.g., from motion or low light).
3. **Smooth Transitions**: A Gaussian ensures smoother detection confidence as conditions change.

## How Gaussian Certainty Factors Work

A Gaussian function can be used to reduce certainty as detection conditions become less ideal. The function is:

$
f(x) = e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$

- $(x)$: The variable, such as distance or angle.
- $(\mu)$: The ideal or expected value, such as the preferred distance.
- $(\sigma)$: Tolerance, or how quickly confidence decreases when moving away from the ideal condition.

### Steps to Apply Certainty Factors

1. **Measure the Conditions**: Get the drone's **altitude**, **camera angle**, and the **distance** to the detected object (people).
2. **Apply the Gaussian Certainty Factor**: Use the Gaussian function to adjust the detection confidence based on how ideal the conditions are.
3. **Combine with YOLO Confidence**: Multiply YOLO's confidence score by the Gaussian certainty factor to adjust the final confidence score.

## Example Code

Here’s how you can implement Gaussian certainty factors in Python:

```python
import numpy as np
import torch

# Load YOLO model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Gaussian function to adjust certainty
def gaussian_certainty(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Example: assume ideal detection happens at 50 meters and with a camera angle of 90 degrees (straight down)
ideal_distance = 50  # in meters
ideal_angle = 90     # in degrees (camera pointing straight down)

# The actual conditions from the drone (e.g., from sensors or calculations)
actual_distance = 70  # in meters
actual_angle = 80     # in degrees

# Standard deviations for how sensitive the confidence is to changes in distance and angle
distance_sigma = 10  # Tolerance for distance variation
angle_sigma = 10     # Tolerance for angle variation

# Apply Gaussian certainty factors
distance_certainty = gaussian_certainty(actual_distance, ideal_distance, distance_sigma)
angle_certainty = gaussian_certainty(actual_angle, ideal_angle, angle_sigma)

# Combine these with YOLO's detection confidence
image = "drone_image.jpg"  # Replace with your actual image
results = model(image)

# Adjust each detection's confidence
for detection in results.pred[0]:
    yolo_confidence = detection[-2].item()  # YOLO's confidence score
    final_confidence = yolo_confidence * distance_certainty * angle_certainty
    detection[-2] = torch.tensor(final_confidence)  # Update the confidence score

# Show updated results
results.show()


