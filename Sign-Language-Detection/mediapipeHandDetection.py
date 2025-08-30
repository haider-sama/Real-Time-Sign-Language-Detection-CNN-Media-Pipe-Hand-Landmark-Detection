import cv2
import mediapipe as mp
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and drawing utilities
handTracker = mp.solutions.hands
handDetector = handTracker.Hands(static_image_mode=False, min_detection_confidence=0.2)
drawing = mp.solutions.drawing_utils
drawingStyles = mp.solutions.drawing_styles

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from the webcam.")
        break

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    imgMediapipe = handDetector.process(frameRGB)

    # Draw hand landmarks if detected
    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,  # Image to draw on
                handLandmarks,  # Detected hand landmarks
                handTracker.HAND_CONNECTIONS,  # Hand connections
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style()
            )

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
