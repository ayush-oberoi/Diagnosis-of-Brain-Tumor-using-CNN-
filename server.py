import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
import numpy as np 
import pandas as pd 
import base64
import io
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow import keras
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from django.shortcuts import render
global graph
global model
global session
import h5py
import requests
from django.conf import settings

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = 'uploads'

'''
def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'gif' else 'jpg')
'''
@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']: #or \
                #file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    model = tf.keras.models.load_model('tumor_prediction.h5', compile=False)
    img = tf.keras.preprocessing.image.load_img('/home/ziel/newproject/test/uploads/'+filename, target_size=(224,224))
    # convert image to an array
    x = image.img_to_array(img)
    # expand image dimensions
    x = preprocess_input(x)
    x = np.expand_dims(x,axis=0)
    rs = model.predict(x)
    rs[0][0]
    rs[0][1]
    if rs[0][0] <= 1 and rs[0][0] >= 0.9:
        result = "Not tumourous"
    else:
        result = "Tumourous"
    return render_template('predict.html', score=rs[0][0], result=result)
'''
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)
'''
if __name__ == '__main__':
    app.run(debug=True)