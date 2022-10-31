from distutils.command.config import config
from re import M, S
import time
import cv2
from threading import Timer
import csv 
from datetime import datetime

import os.path




# global field_flag
# field_flag = True

face_Proto = "models/opencv_face_detector.pbtxt"
face_Model = "models/opencv_face_detector_uint8.pb"

age_Proto = "models/age_deploy.prototxt"
age_Model = "models/age_net.caffemodel"

gender_Proto = "models/gender_deploy.prototxt"
gender_Model = "models/gender_net.caffemodel"

MODEL_MEAN_RANGES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

face_Network = cv2.dnn.readNet(face_Model, face_Proto)
age_Network = cv2.dnn.readNet(age_Model, age_Proto)
gender_Network = cv2.dnn.readNet(gender_Model, gender_Proto)

# vid = cv2.VideoCapture("test.mp4") # use this to load video from file

padding = 20

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes;

gender=[]
age=[]
detected_faces= ""

totalCount=0
# global start_date
# now = datetime.now()
# start_date = now.strftime("%Y-%m-%d %H:%M:%S")

def openCamera():
    global totalCount
    global vid

    now = datetime.now()
    today_date = now.strftime("%d")
    delete_date = 1

    try:
        if int(today_date) == delete_date :
            os.remove('people_records.csv')
    except:
        print('file does not exists')
   
       
    
   
    vid = cv2.VideoCapture(0)
    global gender
    global age
    global detected_faces
    x=True
    while x:
       
        ret, frame = vid.read()
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)
        frameFace, detected_faces = getFaceBox(face_Network, small_frame)
        for dface in detected_faces:
            face = small_frame[
               max(0, dface[1] - padding):min(dface[3] + padding, frame.shape[0] - 1),
                max(0, dface[0] - padding):min(dface[2] + padding, frame.shape[1] - 1)
                ]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_RANGES, swapRB=False)

            gender_Network.setInput(blob)
            genderPreds = gender_Network.forward()
            
            gender.append(GENDER_LIST[genderPreds[0].argmax()]);
           

            age_Network.setInput(blob)
            agePreds = age_Network.forward()
            age.append(AGE_RANGES[agePreds[0].argmax()]); 
            #print(f"Number of people: {len(detected_faces)}");
            
            totalCount=len(detected_faces);

            print(f"Total Number of people= {totalCount}");
            
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            
    
            # field names 
            fields = ['Date','Time','Total People'] 
                
            # data rows of csv file 
            rows = [ [date_str, time_str, totalCount]] 
                
            # name of csv file 
            filename = "people_records.csv"

            file_exists = os.path.isfile('people_records.csv')
            print(file_exists)
                
            # writing to csv file 
            with open(filename, 'a') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                # if field_flag == True:
                if not file_exists:
                    csvwriter.writerow(fields) 
                    # field_flag = False
                    
                # writing the data rows 
                csvwriter.writerows(rows)
            #closeCamera() function release the camera and after 30s it opens camera again and print value
            closeCamera() 
           
            
def closeCamera():
    
   
    vid.release()
    time.sleep(30) 
    openCamera()
           
            
            #for i in range(0,len(detected_faces)):
                
                # print(f'Age: {age[i]} years');
                # print(f'Gender:{gender[i]}')
                # print('--------------');

   

      
#cv2.destroyAllWindows()
          
           
        

#def closeDector():  
   
   
   

    
       
       
            #cv2.putText(frameFace, label, (dface[0], dface[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

       # cv2.putText(frameFace, f"Number of people: {len(detected_faces)}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)  # adjust coordrinates according to your camera resolution


# def readCsvFile():
#     with open('university_records.csv', newline='', encoding='utf-8') as f: 
# 	    reader = csv.reader(f) 
# 	    for row in reader:  
# 	        print(row) 
    
def removeCsv():
    os.remove('people_records.csv')