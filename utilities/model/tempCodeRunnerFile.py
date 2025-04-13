with torch.no_grad():
                outputs = model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
            gesture_name = class_names.get(predicted_class, "Unknown")
            # Get position to display the gesture (using wrist landmark)
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, gesture_name, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Predicted Gesture: {gesture_name}")