import cv2
import os

# Define the folder path where images will be saved (data/rotate)
gesture_name = "test"
output_folder = os.path.join("utilities/training_data", gesture_name)

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize a counter to name captured images
img_counter = 0

print("Press 'c' to capture an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display instructions on the frame
    cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Capture Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Build the filename and save the image
        img_name = f"{gesture_name}_{img_counter:03d}.jpg"
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, frame)
        print(f"Captured {save_path}")
        img_counter += 1
    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
