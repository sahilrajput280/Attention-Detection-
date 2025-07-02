import cv2
import os
from datetime import datetime

# Set the folder where you want to save the photo
save_folder = r"D:\AIML\DATA\train\not_attentive"
os.makedirs(save_folder, exist_ok=True)

# Use the default camera (0)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 's' to capture and save the image. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the camera feed
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        # Generate a unique filename using the current timestamp
        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(save_folder, filename)
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")

    elif key == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
