import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
from model import ResNet9,to_device,get_default_device,predict_image,selectModel
import torch
import torchvision.transforms as tt
import os
from PyQt5.QtWidgets import *
import torchvision.models as models
import torch.nn as nn
run_flag = True
x,y,w,h=275,30,250,250
fixed=False
iterations1=1
iterations2=1
km=128
def setFlag():
    global run_flag
    run_flag=False

def move(u=False,d=False,l=False,r=False,auto=False):
    global x,y,fixed
    move=10
    if u:
        y=y-move
    if d:
        y=y+move
    if l:
        x=x-move
    if r:
        x=x+move
    if auto:
        fixed= not fixed
def erode(it):
    global iterations1
    iterations1=it

def dialate(it):
    global iterations2
    iterations2=it

def kmeans(k):
    global km
    km=k

def detect(change_pixmap_signal1,change_pixmap_signal2,tl1,tl2,mod="Indian"):
    global x,y,w,h
    global run_flag
    global fixed,iterations1,iterations2
    global km
    run_flag=True
    selectModel(mod)
    target_num=28
    device = get_default_device()
    model = models.mobilenet_v2()
    in_features = model._modules['classifier'][-1].in_features
    model._modules['classifier'][-1] = nn.Linear(in_features, target_num, bias=True)
    model = to_device(model, device)
    if mod=="Indian":
        target_num=36
        device = get_default_device()
        model = models.mobilenet_v2()
        in_features = model._modules['classifier'][-1].in_features
        model._modules['classifier'][-1] = nn.Linear(in_features, target_num, bias=True)
        model = to_device(model, device)
        model.load_state_dict(torch.load(os.path.join("..","models","MobileNet_V2Indian70img.pth"),map_location=torch.device('cpu')))
        model.eval()
    else:
        model.load_state_dict(torch.load(os.path.join("..","models","MobileNet_V2ASLEroded.pth"),map_location=torch.device('cpu')))
        model.eval()

    pred=[]
    transform = tt.Compose([tt.ToTensor(),tt.Resize(size=(128,128))])
    kernel1 = np.ones((2,2),np.uint8)
    minValue = 70
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
 
    fixed=True
    offset=20
    imgSize = 400
    while run_flag:
        try: 
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                
                if not fixed:
                    x, y, w, h = hand['bbox']

                if(w<250):
                    w=250
                if(h<250):
                    h=250
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
                imgCrop = imgOutput[y - offset:y + h + offset, x - offset-50:x + w + offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                    gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(5,5),2)
                    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
                    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

                    res = cv2.dilate(res,kernel1,iterations1)
                    res = cv2.erode(res,kernel1,iterations2)

                    # if(mod=="Indian"):
                    #     kernel = np.ones((5,5),np.uint8)
                    #     res = cv2.erode(res,kernel,iterations = 1)

                    imgtensor= transform(res)
                    predicted=predict_image(imgtensor, model)
                    tl1.setText(predicted)
                    pred.append(predicted)
                    if len(pred)>=50:
                        if pred.count(max(set(pred), key = pred.count))>35:
                            tl2.setText(tl2.text()+max(set(pred), key = pred.count))
                            pred.clear()
                        else:
                            pred.clear()
                    

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    img=imgWhite
                    #img=cv2.addWeighted(img, 1,img,2,-10)
                    if(km<101):
                        img=np.float32(img)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) #criteria
                        retval, labels, centers = cv2.kmeans(img, km, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS) 
                        centers = np.uint8(centers) # convert data into 8-bit values 
                        segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
                        img = segmented_data.reshape((img.shape))
 
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(5,5),2)

                    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
                    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
                    res= cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                    
                    res = cv2.dilate(res,kernel1,iterations=iterations2)
                    res = cv2.erode(res,kernel1,iterations=iterations1)

                    imgtensor= transform(res)
                    predicted=predict_image(imgtensor, model)
                    tl1.setText(predicted)
                    pred.append(predicted)

                    if len(pred)>=30:
                        if pred.count(max(set(pred), key = pred.count))>28:
                            if max(set(pred), key = pred.count) =="Nothing":
                                tl1.setText("Place Hand in box")
                            elif max(set(pred), key = pred.count) =="Space":
                                tl2.setText(tl2.text()+" ")
                            else:
                                tl2.setText(tl2.text()+max(set(pred), key = pred.count))
                                
                            pred.clear()
                        else:
                            pred.clear()
                    
                change_pixmap_signal2.emit(res)
                cv2.rectangle(imgOutput, (x-offset-50, y-offset),
                            (x + w+offset, y + h+offset), (255, 0, 255), 4)
            
                change_pixmap_signal1.emit(imgOutput)
            cv2.waitKey(1)
        except Exception as e: 
            print(e)