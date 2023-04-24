import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from model import ResNet9,to_device,get_default_device,predict_image
import torch
import torchvision.transforms as tt
import torchvision.models as models
import mediapipe as mp
import os
import torch.nn as nn
def fun(change_pixmap_signal1,change_pixmap_signal2,run_flag,tl1,tl2):
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    pred=[]


    # def SegmentNorm(img):
    #     BG_COLOR = (0,0,0)
    #     with mp_selfie_segmentation.SelfieSegmentation(
    #         model_selection=0) as selfie_segmentation:
    #         image = img
    #         results = selfie_segmentation.process(image)
    #         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.2
    #         bg_image = np.zeros(image.shape, dtype=np.uint8)
    #         bg_image[:] = BG_COLOR
    #         imgbremoved= np.where(condition, image, bg_image)

    #         transform = tt.ToTensor()
    #         imgbremoved= transform(imgbremoved)

    #         mean=imgbremoved.mean([1,2])
    #         std=imgbremoved.std([1,2])
    #         stats=(mean,std)
    #         transform = tt.Compose([tt.Normalize(*stats)])
    #         imgbremoved=transform(imgbremoved)
    #     return imgbremoved

    transform = tt.Compose([tt.ToTensor(),tt.Resize(size=(128,128))])

    target_num=28
    device = get_default_device()
    model=models.resnet18(pretrained=False)
    in_features = model._modules['fc'].in_features
    model._modules['fc'] = nn.Linear(in_features, target_num, bias=True)
    model = to_device(model, device)
    model.load_state_dict(torch.load("..\\models\\resnet19.pth",map_location=torch.device('cpu')))
    model.eval()
    minValue = 50

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    offset=20
    imgSize = 400
    while run_flag:
        try: 
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
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

                    gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(5,5),2)

                    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
                    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

                    # kernel = np.zeros((3,3), np.uint8) 
                    # # img_erosion = cv2.erode(res, kernel, iterations=2) 
                    # res = cv2.dilate(res, kernel, iterations=5) 

                    # kernel = np.ones((1,1), np.uint8) 
                    # img_erosion = cv2.erode(res, kernel, iterations=2) 
                    # res = cv2.dilate(img_erosion, kernel, iterations=5) 

                    res= cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
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
                    

                # crop= imgseg.permute(1,2,0).cpu().detach().numpy()
                    
                change_pixmap_signal2.emit(res)
                cv2.rectangle(imgOutput, (x-offset-50, y-offset),
                            (x + w+offset, y + h+offset), (255, 0, 255), 4)
            
                change_pixmap_signal1.emit(imgOutput)
            cv2.waitKey(1)
        except: 
            pass
