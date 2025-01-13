import cv2 
import pandas as pd # data manupilation and analysis
from ultralytics import YOLO  # detect object
import cvzone  # related to computer vision task ( boxes and names)
import numpy as np  # for mathematical operations


# model setup (pre trained model)
model=YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        
# optional
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB) # setup the call back function to be executed 
cap=cv2.VideoCapture('he2.mp4')

# class labels (contains the list of object to detect)
my_file = open(r"C:\Users\DELL\Downloads\yolov8helmetdetection-main\yolov8helmetdetection-main\coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0


# vedio capture and processing (main part)  look every thirt page of the book

while True:    
    ret,frame = cap.read() # read frame from vedio
    if not ret:  # if frame is read ret is true otherwise video is end
        break
    count += 1  # use to control frequency  (this entire process for third time of vedio) optimize performance
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500)) # resizing the frame for better processing speed
   

    # object detection 
    results=model.predict(frame)  #analyse and return object 
 #   print(results)
    a=results[0].boxes.data # extracting bounding boxes , boxes as atribute
    px=pd.DataFrame(a).astype("float") # pandas data frame .astype confirms all the numbers 
#    print(px)
    
    list=[] #
    
    # bounding box
    for index,row in px.iterrows():  #  row represent data
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
 
      
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2) # display frame
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) # display text


    cv2.imshow("RGB", frame) 
    if cv2.waitKey(1)&0xFF==27:  # if press ESC key then code break the loop
        break
cap.release()  
cv2.destroyAllWindows()
