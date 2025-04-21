"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Gesture Image Capture Tool
------------------
This script captures and saves images from a webcam for training computer vision models
to recognize specific gestures or poses. It's particularly useful for creating custom
datasets for machine learning applications.

Usage:
    - Run the script
    - Press 'c' to capture an image
    - Press 'q' to quit
    - Captured images are saved in the specified output folder with sequential numbering
"""

# Import required libraries
import cv2  # OpenCV library for computer vision operations
import os   # Operating system module for file and directory operations

# Configuration settings
gesture_name = "test"  # Name of the gesture being captured
output_folder = os.path.join("utilities/training_data", gesture_name)  # Path where images will be saved

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# Initialize video capture device
# The parameter 0 selects the default webcam; use different numbers for additional cameras
cap = cv2.VideoCapture(0)

# Initialize counter for naming image files sequentially
img_counter = 0

# Display instructions to the user
print("Press 'c' to capture an image, 'q' to quit.")

# Main capture loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame was successfully captured
    if not ret:
        print("Failed to grab frame.")
        break

    # Add on-screen instructions to the frame
    cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Capture Gesture", frame)

    # Wait for key press and process user input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        # Generate a filename with sequential numbering (e.g., test_000.jpg, test_001.jpg)
        img_name = f"{gesture_name}_{img_counter:03d}.jpg"
        save_path = os.path.join(output_folder, img_name)
        
        # Save the current frame as an image
        cv2.imwrite(save_path, frame)
        print(f"Captured {save_path}")
        
        # Increment the counter for the next image
        img_counter += 1
    
    elif key == ord('q'):
        # Exit the loop if 'q' is pressed
        break

# Clean up resources when done
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows