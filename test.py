import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from keras.models import load_model

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the hand sign model
hand_classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Load the emotion detection model
emotion_model = load_model("Model2/keras_model.h5")
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Initialize the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)
overlay_width = 437 - 19
overlay_height = 406 - 88

imgBackground = cv2.imread('Data/ssbg.png')

# Set parameters for hand sign detection
offset = 20
imgSize = 300
labels = [
    "I'm in Danger",
    "Signal for help",
    "Call Police",
    "Call Ambulance",
    "Visit My Home",
    "Call me",
    "Losing",
    "I am in pain",
    "I am okay",
    "Thief"
]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    # Hand detection
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop is not empty
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = hand_classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = hand_classifier.getPrediction(imgWhite, draw=False)

            # Display hand sign prediction
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Face detection
    # Face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]  # cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Preprocess the image for emotion detection
        img_pixels = roi_gray.astype('float32') / 255.0
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        cv2.putText(imgOutput, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #imgBackground = cv2.resize(imgBackground, (imgOutput.shape[1], imgOutput.shape[0]))

    imgOutput_resized = cv2.resize(imgOutput, (overlay_width, overlay_height))

    # Overlay imgOutput_resized onto imgBackground
    imgBackground[88:88 + overlay_height, 19:19 + overlay_width] = imgOutput_resized

    #imgBackground[162:162 + imgOutput.shape[0], 55:55 + imgOutput.shape[1]] = imgOutput
    cv2.imshow("Image", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from keras.models import load_model
import matplotlib.pyplot as plt

# Load models
hand_detector = HandDetector(maxHands=1)
hand_classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
emotion_model = load_model("Model2/keras_model.h5")

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera setup
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

# Define labels for hand signs
labels = [
    "I'm in Danger",
    "Signal for help",
    "Call Police",
    "Call Ambulance",
    "Visit My Home",
    "Call me",
    "Losing",
    "I am in pain",
    "I am okay",
    "Thief"
]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    # Hand sign detection
    hands, img = hand_detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = hand_classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = hand_classifier.getPrediction(imgWhite, draw=False)

            # Display prediction on output
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Emotion detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.32, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Add batch dimension
        img_pixels = np.expand_dims(roi_gray, axis=0)

        img_pixels = np.array(img_pixels, dtype=np.float32)
        img_pixels /= 255

        predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(imgOutput, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Combined Detection", imgOutput)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from keras.models import load_model
import matplotlib.pyplot as plt

# Load models
hand_detector = HandDetector(maxHands=1)
hand_classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
emotion_model = load_model("Model2/keras_model.h5")

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera setup
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    # Hand sign detection
    hands, img = hand_detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop is not empty
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
            # ... (Rest of the hand sign detection code)

    # Emotion detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.32, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = np.array(roi_gray, dtype=np.float32)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(imgOutput, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Combined Detection", imgOutput)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  '''
