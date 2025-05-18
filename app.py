import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import time


# Load OpenCV Pre-trained Age Model
AGE_PROTO = "Models/age_deploy.prototxt"
AGE_MODEL = "Models/age_net.caffemodel"
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# Face Detector
mtcnn = MTCNN(keep_all=True)

def predict_age(face_img):
    try:
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     (78.426, 87.768, 114.895), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        max_idx = preds[0].argmax()
        confidence = preds[0][max_idx]
        return AGE_BUCKETS[max_idx], confidence
    except:
        return "N/A", 0.0

class AgeEstimator(VideoTransformerBase):

    def __init__(self):
        self.last_prediction_time = 0
        self.prediction_interval = 1  # seconds
        self.last_age = "N/A"
        self.last_confidence = 0.0


    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        boxes, _ = mtcnn.detect(img)

        current_time = time.time()

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                if current_time - self.last_prediction_time > self.prediction_interval:
                    age, confidence = predict_age(face)
                    self.last_age = age
                    self.last_confidence = confidence
                    self.last_prediction_time = current_time

                confidence_percent = f"{self.last_confidence * 100:.1f}%"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'Age: {self.last_age} ({confidence_percent})', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)
        return img

st.title("Age Estimator")

webrtc_streamer(key="age_estimation", video_processor_factory=AgeEstimator,
                media_stream_constraints={"video": True, "audio": False})
