import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

window = tk.Tk()
window.title("Smart Attendance System Using Face Recognition")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
 
window.geometry('1280x720')
window.configure(background='grey')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message1 = tk.Label(window, text="Madhav Institute of Technology and Science, Gwalior", fg="black", bg="grey", font=('times', 30, 'bold'))
message1.place(x=350, y=10)

message2 = tk.Label(window, text="Major Project", fg="black", bg="grey", font=('times', 20, 'bold'))
message2.place(x=700, y=70)

message3 = tk.Label(window, text="on", fg="black", bg="grey", font=('times', 15, 'bold'))
message3.place(x=770, y=100)

message4 = tk.Label(window, text="Smart Attendance System Using Face Recognition", fg="black", bg="grey", font=('times', 30, 'italic bold underline'))
message4.place(x=370, y=130)

lbl = tk.Label(window, text="Roll Number :", width=20, height=2, bg="grey", font=('times', 15, ' bold ') )
lbl.place(x=400, y=225)

txt = tk.Entry(window,width=20  ,bg="white" ,font=('times', 15, ' bold '))
txt.place(x=700, y=235)

lbl2 = tk.Label(window, text="Name :",width=20  ,bg="grey"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=400, y=325)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,font=('times', 15, ' bold ')  )
txt2.place(x=700, y=335)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,bg="grey"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl3.place(x=400, y=425)

message = tk.Label(window, text="" ,bg="grey"  ,width=20  ,height=4, activebackground = "yellow" ,font=('times', 15, ' bold '))
message.place(x=650, y=410)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,bg="grey"  ,height=2 ,font=('times', 15, ' bold '))
lbl3.place(x=400, y=650)
message5 = tk.Label(window, text="" ,bg="grey", activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold '))
message5.place(x=600, y=650)

    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
 
def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    if(is_number(Id) and name != ""):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>10:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Trained for\nRoll Number: " + Id +"\nName: "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        TrainImages()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    # res = "Image Trained"#+",".join(str(f) for f in Id)
    # message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 60):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            # if(conf > 50):
            #     noOfFile=len(os.listdir("ImagesUnknown"))+1
            #     cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)
    res = "Attendance Mraked"
    message5.configure(text= res)


takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=450, y=525)
trackImg = tk.Button(window, text="Take Attendance", command=TrackImages  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=525)

 
window.mainloop()