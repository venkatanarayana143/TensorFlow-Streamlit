import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
import requests


header_container = st.container()
stats_container = st.container()

with header_container:

  st.title("Omdena - Ahmedabad Chapter  Anomaly Detection on Mars ")
  st.sidebar.title("Omdena - Ahmedabad Chapter: Anomaly Detection on Mars")
  st.sidebar.header("The List of Anomalies are as follow:")
  
  st.sidebar.subheader("Class 1: Craters")
  url ='https://www.marsartgallery.com/images/martiancraterplaxco.jpg'
  image = Image.open(requests.get(url, stream=True).raw)
  st.sidebar.image(image, caption='Craters on Mars')
  st.sidebar.subheader("Craters are caused when a bolide collides with a planet.The Martian surface contains thousands of impact craters because, unlike Earth, Mars has a stable crust, low erosion rate, and no active sources of lava")

  st.sidebar.subheader("Class 7: Spiders")
  url ='https://dailyzooniverse.files.wordpress.com/2013/09/martian-spiders.jpg'
  image = Image.open(requests.get(url, stream=True).raw)
  st.sidebar.image(image, caption='Spiders on Mars')
  st.sidebar.subheader("Spiders are actually topological troughs formed when dry ice directly sublimates to a gas")

  

  model = tf.keras.models.load_model("mdl_wts.hdf5")
  ### load file
  uploaded_file = st.file_uploader("Choose a image file", type="jpg")

  map_dict = {0: 'dog',
              1: 'horse',
              2: 'elephant',
              3: 'butterfly',
              4: 'chicken',
              5: 'cat',
              6: 'cow',
              7: 'sheep',
              8: 'spider',
              9: 'squirrel'}


  if uploaded_file is not None:
      # Convert the file to an opencv image.
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)
      opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
      resized = cv2.resize(opencv_image,(224,224))
      # Now do something with the image! For example, let's display it:
      st.image(opencv_image, channels="RGB")

      resized = mobilenet_v2_preprocess_input(resized)
      img_reshape = resized[np.newaxis,...]

      Genrate_pred = st.button("Generate Prediction")    
      if Genrate_pred:
          prediction = model.predict(img_reshape).argmax()
          st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
