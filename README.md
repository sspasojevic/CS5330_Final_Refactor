# 3D Object Manipulation via Hand Gesture Recognition

## Team Members

- Sara Spasojevic
- Matej Zecic
- Benjamin Northrop

## Project Description

This project implements a real-time 3D object manipulation system using computer vision-based hand gesture recognition. Users can interact with 3D objects in a virtual environment using natural hand gestures captured by a webcam. The application recognizes specific hand gestures to control translation, rotation, and scaling of 3D objects rendered in the scene.

The system uses MediaPipe for hand landmark detection and a custom-trained PyTorch model for gesture classification. This approach allows for intuitive interactions with virtual objects without requiring specialized hardware or controllers.

## Demo Videos

- [Demo Video](https://www.youtube.com/watch?v=ZyzjnK2vaBM)

## Setup Instructions

### System Requirements

- Python 3.9-3.11 (3.11 recommended for MediaPipe compatibility)
- Webcam access
- Modern GPU recommended for smooth performance

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/3D-object-manipulation.git
   cd CS5330_Final_Refactor

   ```

2. **Create a virtual environment**

   ```bash
   # On Windows
   python3.11 -m venv venv

   # On macOS/Linux
   python3.11 -m venv venv

   ```

3. **Activate the virtual environment**

   ```bash
   # On Windows
    .\venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
   ```

4. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python scene.py
   ```

### Usage instruction

1. Position yourself in front of your webcam
2. The application will detect your hand and display landmarks on the video feed
3. Use the following gestures to manipulate the 3D object:
   - Grab & Move (Left/Right Hand): Translates the object
   - Scale (Left/Right Hand): Increases or decreases the object size
   - Swipe (Left/Right Hand): Rotates the object
   - Hold (Left/Right Hand): Maintains current position/orientation

### Project Structure

- program/: Contains the main application code
  - gesture_recognizer.py: Handles gesture detection and classification
  - state_changer.py: This module provides a communication mechanism between gesture recognition and 3D object manipulation.
  - orbit_camera.py: Provides a camera system for 3D applications that orbits around a target position.
  - scene_object.py: 3D object representation for rendering in a moderngl-based graphics system.
  - shader_program.py: Offers a convenient interface for working with multiple shader
    programs in a ModernGL-based rendering system.
- utilities/: Helper modules and resources
  - model/: Contains the trained gesture classification model
  - render_data/: 3D models and textures
- scene.py: Entry point for the application

### Acknowledgements

- MediaPipe for hand detection and landmark tracking
- PyTorch for the gesture classification model
- ModernGL for 3D rendering
  License
