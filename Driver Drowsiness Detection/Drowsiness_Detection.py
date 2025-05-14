
import time
import cv2
import dlib
import numpy as np
from imutils import face_utils
from pygame import mixer
import os
from twilio.rest import Client
from scipy.spatial import distance

# Twilio API configuration
TWILIO_ACCOUNT_SID = '--------------'
TWILIO_AUTH_TOKEN = '---------------'
TWILIO_PHONE_NUMBER = '----------'
RECIPIENT_PHONE_NUMBER = '----------'

# Initialize mixer for playing sound
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load pre-trained facial landmark predictor
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set parameters
thresh = 0.25  # Common threshold value
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0  # To keep track of the frame number
closed_eye_frame_count = 0  # To count closed eye frames
start_time = 0  # Initialize start_time
alert_duration = 2  # Duration to detect drowsiness (in seconds)
output_directory = "closed_eye_frames"  # Directory to save closed eye frames
frame_interval = 50  # Interval for saving frames

# Create directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

overall_frame_count = 0  # To count the total number of frames processed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    eyes_closed = False  # Flag to track if eyes are closed

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < thresh:
            if start_time == 0:
                start_time = time.time()
            elif time.time() - start_time >= alert_duration:
                eyes_closed = True
        else:
            start_time = 0  # Reset start_time if eyes are open

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if eyes_closed:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "ALERT!", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()

            try:
                # Send SMS alert using Twilio
                message = client.messages.create(
                    body="Drowsiness detected! Please take a break.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=RECIPIENT_PHONE_NUMBER
                )
                print("SMS sent successfully!")
            except Exception as e:
                print("Error sending SMS:", str(e))

            # Save the frame where eyes are detected as closed
