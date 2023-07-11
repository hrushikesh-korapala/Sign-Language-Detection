from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import time
from playground.camera import VideoCamera

# os.environ["cuda_visible_devices"]="-1"

# Create your views here.
# request handler
# request -> response


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sign_model=models.load_model("./action.h5")

def say_hello(request):
    # return HttpResponse('Hello World')
    return render(request,'hello.html')


def gen(camera):
	while True:
		frame = camera.get_frame()
        
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def get_prediction(request):
    pic=np.zeros((1,30,1662))
    y=sign_model.predict(pic)
    return HttpResponse(y)
    