from flask import Flask,render_template,url_for,request
from flask_material import Material
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objs as go
import json
from flask import flash
import tkinter as tk
from tkinter import ttk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time




#

# ML Pkg
window = tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("Face Recgnition System")
window.configure(background='#3813a0')

############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

###################################################################################


###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

###################################################################################







def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids




    
    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }



app = Flask(__name__)
Material(app)
app.secret_key="dont tell any one"
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Registration.html')
def Registration():
    return render_template("Registration.html")

@app.route('/Attendance.html')
def Attendance():
    return render_template("Attendance.html")


@app.route('/',methods=["POST"])
def login():
    if request.method == 'POST':
        username = request.form['id']
        password = request.form['pass']
        if username=='admin' and password=='admin':
            return render_template("main.html")
        else:
            flash("wrong password")
            return render_template("index.html")


@app.route('/analyze',methods=["POST"])

def analyze():
    if request.method == 'POST':
        if request.form['submit'] == 'Take_Images':
            Id = request.form['Enter_ID']
            name = request.form['name']
            email = request.form['email']
            course=request.form['course']
            year=request.form['year']
            sec=request.form['year']
            check_haarcascadefile()
            columns = ['SERIAL NO.', '', 'ID', '', 'NAME', '', 'EMAIL', '', 'course', '', 'year', '', 'sec']
            assure_path_exists("StudentDetails/")
            assure_path_exists("TrainingImage/")
            serial = 0
            exists = os.path.isfile("StudentDetails\StudentDetails.csv")
            data=pd.read_csv("StudentDetails\StudentDetails.csv")
            data=data['ID']
            data=data.dropna()
            print("data",data)
            Id1=int(Id)
            names = data.tolist()
            [int(i) for i in names]

            for i in names:
                if Id1==i:
                    msg1="ID already Exist"
                    return render_template('Registration.html', res=msg1)

            
            if exists:
                with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for l in reader1:
                        serial = serial + 1
                serial = (serial // 2)
                csvFile1.close()
            else:
                with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(columns)
                    serial = 1
                csvFile1.close()
            if ((name.isalpha()) or (' ' in name)):
                cam = cv2.VideoCapture(0)
                harcascadePath = "haarcascade_frontalface_default.xml"
                detector = cv2.CascadeClassifier(harcascadePath)
                sampleNum = 0
                while (True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # incrementing sample number
                        sampleNum = sampleNum + 1
                        # saving the captured face in the dataset folder TrainingImage
                        cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                                    gray[y:y + h, x:x + w])
                        # display the frame
                        cv2.imshow('Taking Images', img)
                    # wait for 100 miliseconds
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    # break if the sample number is morethan 100
                    elif sampleNum > 100:
                        break
                cam.release()
                cv2.destroyAllWindows()
                res = "Images Taken for ID : " + Id
                row = [serial, '', Id, '', name, '', email, '', course, '', year, '', sec]
                with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
            else:
                if (name.isalpha() == False):
                    res = "Enter Correct name"
            return render_template('Registration.html', res=res)
        
        if request.form['submit'] == 'Save_Profile':
            check_haarcascadefile()
            assure_path_exists("TrainingImageLabel/")
            recognizer = cv2.face_LBPHFaceRecognizer.create()
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            faces, ID = getImagesAndLabels("TrainingImage")
            try:
                recognizer.train(faces, np.array(ID))
            except:
                mess._show(title='No Registrations', message='Please Register someone first!!!')
                return
            recognizer.save("TrainingImageLabel\Trainner.yml")
            res = "Profile Saved Successfully"
            res1= "Total Registrations till now  : "+ str(ID[0])
            return render_template('Registration.html', res=res,res1=res1)

        if request.form['submit'] == 'Take_Attendance':
            check_haarcascadefile()
        assure_path_exists("FaceRecognition/")
        assure_path_exists("StudentDetails/")
        msg = ''
        i = 0
        j = 0
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
        if exists3:
            recognizer.read("TrainingImageLabel\Trainner.yml")
        else:
            mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
            return
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', '', 'Name', '', 'Email', '', 'Date', '', 'Time', '', 'Course', '', 'Year', '', 'Section']
        exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
        if exists1:
            df = pd.read_csv("StudentDetails\StudentDetails.csv")
            print(df)
        else:
            mess._show(title='Details Missing', message='Students details are missing, please check!')
            cam.release()
            cv2.destroyAllWindows()
            window.destroy()
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 60):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                    ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                    ee = df.loc[df['SERIAL NO.'] == serial]['EMAIL'].values

                    xx = df.loc[df['SERIAL NO.'] == serial]['Course'].values
                    yy = df.loc[df['SERIAL NO.'] == serial]['Year'].values
                    zz = df.loc[df['SERIAL NO.'] == serial]['Section'].values

                    ID = str(ID)
                    ID = ID[1:-1]
                    bb = str(aa)
                    bb = bb[2:-2]
                    ff= str(ee)
                 
                    ff= ff[2:-2]


                    xx1 = str(xx)
                    xx1 = xx1[2:-2]
                    print("xx1",xx1)
                    yy1 = str(yy)
                    yy1 = yy1[1:-1]
                    print("yy1",yy1)
                    zz1= str(zz)
                    
                    zz1= zz1[1:-1]
                    print("zz1",zz1)
                    
                    attendance = [str(ID), '', bb, '', ff, '', str(date), '', str(timeStamp), '', xx1, '', yy1, '', zz1]
                

                else:
                    Id = 'Unknown'
                    bb = str(Id)
                cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
            cv2.imshow('Taking Face Image', im)
            if (cv2.waitKey(1) == ord('q')):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        exists = os.path.isfile("FaceRecognition\FaceRecognition_" + date + ".csv")
        if exists:
            with open("FaceRecognition\FaceRecognition_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(attendance)
            csvFile1.close()
        else:
            with open("FaceRecognition\FaceRecognition_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(col_names)
                writer.writerow(attendance)
            csvFile1.close()
        csvFile1.close()
        cam.release()
        cv2.destroyAllWindows()
        return render_template('Attendance.html')

if __name__ == '__main__':
    app.run(debug=True)
