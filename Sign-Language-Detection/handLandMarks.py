import os
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Initialize the MediaPipe Hands class for hand tracking
handTracker = mp.solutions.hands
handDetector = handTracker.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2
)  # Configure the MediaPipe Hands instance for detecting hands

# Dataset folder path
DATA_FOLDER = "../datasets/asl_alphabet_test"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Initialize variables
coordinates = []  # List to store data for all characters
index = 0

# Process images in the dataset folder
for file in os.listdir(DATA_FOLDER):
    for imgPath in os.listdir(os.path.join(DATA_FOLDER, str(file))):
        fullImgPath = os.path.join(DATA_FOLDER, file, imgPath).replace("\\", "/")
        
        try:
            image = Image.open(fullImgPath)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to load image from {fullImgPath}: {e}")
            continue  # Skip this iteration if the image failed to load

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgMediapipe = handDetector.process(imgRGB)

        x_Coordinates = []
        y_Coordinates = []
        z_Coordinates = []

        if imgMediapipe.multi_hand_landmarks:
            for handLandmarks in imgMediapipe.multi_hand_landmarks:
                data = {
                    "CHARACTER": file,
                    "GROUPVALUE": index
                }

                for lm in handLandmarks.landmark:
                    x_Coordinates.append(lm.x)
                    y_Coordinates.append(lm.y)
                    z_Coordinates.append(lm.z)

                for i, landmark in enumerate(handTracker.HandLandmark):  # Apply Min-Max normalization
                    lm = handLandmarks.landmark[i]
                    data[f"{landmark.name}_x"] = lm.x - min(x_Coordinates)
                    data[f"{landmark.name}_y"] = lm.y - min(y_Coordinates)
                    data[f"{landmark.name}_z"] = lm.z - min(z_Coordinates)

                coordinates.append(data)

    index += 1

# Convert collected data to a DataFrame and save to Excel
try:
    df = pd.DataFrame(coordinates)
    excel_path = "asl_alphabet_testing_data.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Data successfully saved to {excel_path}")
except Exception as e:
    print(f"Failed to save data to Excel: {e}")
