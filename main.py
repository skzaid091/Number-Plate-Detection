import opencv-python-headless as cv2
import os
from PIL import Image
import numpy as np
import easyocr
import streamlit as st
import time
import string


config_file = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'models/frozen_inference_graph.pb'
plate_cascade = cv2.CascadeClassifier('models/haarcascade_russian_plate_number.xml')

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

classLabels = []
file_name = 'models/labels.txt'
with open(file_name, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')


def extract_number_plate(img):
    detected_number_plate = []
    if img is None:
        print("Error: Unable to read image.")
        return
    classIndex, confidence, bbox = model.detect(img, confThreshold=0.6)
    if len(classIndex) == 0:
        st.write("Image is not Supporting")
        return
    else:
        for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            x, y, w, h = boxes
            car_img = img[y:y + h, x:x + w]
            convert_to_gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(convert_to_gray, scaleFactor=1.1, minNeighbors=4)
            plate_img = convert_to_gray[y:y + h + 20, x - 25:x + w]
            if plate_img.size == 0:
                print("Plate image is empty. Skipping...")
                continue
            reader = easyocr.Reader(['en'])
            results = reader.readtext(plate_img)

            # Replace 'O' with '0' in the detected text
            for i in range(len(results)):
                results[i] = (results[i][0], results[i][1].replace('O', '0'), results[i][2])
            
            # Replace 'I' with '1' in the detected text
            for i in range(len(results)):
                results[i] = (results[i][0], results[i][1].replace('I', '1'), results[i][2])
                
            for (bbox, text, prob) in results:
                if prob > 0.20:
                    detected_number_plate.append(text)
        return detected_number_plate


def process_image(uploaded_file):
    image1 = Image.open(uploaded_file)
    image2 = np.array(image1)
    
    start_time = time.time()
    data = extract_number_plate(image2)

    if len(data) == 0:
        st.header("Model is NOT Working on this IMAGE")
        return
    indian_vehicle_registration_codes = [
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CH",  # Chandigarh
    "CG",  # Chhattisgarh
    "DD",  # Dadra and Nagar Haveli and Daman and Diu
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HR",  # Haryana
    "HP",  # Himachal Pradesh
    "JH",  # Jharkhand
    "KA",  # Karnataka
    "KL",  # Kerala
    "LD",  # Lakshadweep
    "MP",  # Madhya Pradesh
    "MH",  # Maharashtra
    "MN",  # Manipur
    "ML",  # Meghalaya
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PY",  # Puducherry
    "PB",  # Punjab
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TS",  # Telangana
    "TR",  # Tripura
    "UP",  # Uttar Pradesh
    "UK",  # Uttarakhand
    "WB"   # West Bengal
    ]
    if data is None:
         exit()

    end_time = time.time()
    execution_time = end_time - start_time

    print("DATA : ", data)

    if len(data) > 1:
        if data[0] != data[1]:
            plate_text = ' '.join(data)
        else:
            plate_text = data[0]
    else:
        plate_text = data[0]

    # Using string module and str.translate() method
    translator = str.maketrans('', '', string.punctuation)
    plate_text = plate_text.translate(translator)

    flag = 0

    for state_code in indian_vehicle_registration_codes:
        if state_code in plate_text:
            flag = 1
            break
        else:
            flag = 0

    if flag == 0:
        st.write("Not an Indian Vehicle OR NUMBER PLATE NOT DETECTED PROPERLY")
        print("plate is : ", plate_text[:].upper())
        return
    
    if len(plate_text) > 2 and plate_text[2] != ' ':
        plate_text = plate_text[:2] + ' ' + plate_text[2:]
    st.markdown(f"## {plate_text.upper()}")
    st.write("Execution time:", execution_time, "seconds")    

st.title("ANPR")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    process_image(uploaded_file)
