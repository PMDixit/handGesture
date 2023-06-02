from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import handDetect
import gtts  
from playsound import playsound 
import os
import threading
import pygame
class VideoThread(QThread):
    change_pixmap_signal1 = pyqtSignal(np.ndarray)
    change_pixmap_signal2 = pyqtSignal(np.ndarray)

    def __init__(self,tl1,tl2):
        super().__init__()
        self.model="Indian"
        self.run_flag=True
        self.tl1=tl1
        self.tl2=tl2

    def run(self):
        while self.run_flag:
            handDetect.detect(self.change_pixmap_signal1,self.change_pixmap_signal2,self.tl1,self.tl2,mod=self.model)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self.run_flag=True
        self.wait()
        self.terminate()
    
    def changeModel(self,model):
        self.model=model
        handDetect.setFlag()


class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("self")
        self.setWindowTitle("Hand Gesture Recognition")
        self.resize(1024, 732)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.setFont(font)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.image_label1 = QtWidgets.QLabel(self.centralwidget)
        self.image_label1.setGeometry(QtCore.QRect(10, 20, 600, 400))
        self.image_label1.setObjectName("image_label1")
        self.image_label1.setStyleSheet("border: 3px solid black; padding-left:45px")

        self.image_label2 = QtWidgets.QLabel(self.centralwidget)
        self.image_label2.setGeometry(QtCore.QRect(615, 20, 400, 400))
        self.image_label2.setObjectName("image_label2")
        self.image_label2.setStyleSheet("border: 3px solid black;padding-left:20px")
        #----------------------------------------------------------------------------------------------


        self.slidererode = QtWidgets.QSlider(self.centralwidget)
        self.slidererode.setFocusPolicy(Qt.StrongFocus)
        self.slidererode.setGeometry(QtCore.QRect(340,680,100,50))
        self.slidererode.setOrientation(QtCore.Qt.Horizontal)
        self.slidererode.setTickPosition(QSlider.TicksBothSides)
        self.slidererode.setTickInterval(20)
        self.slidererode.setSingleStep(5)
        self.slidererode.valueChanged.connect(self.changeErode)

        self.labelerode = QtWidgets.QLabel(self.centralwidget)
        self.labelerode.setGeometry(QtCore.QRect(340, 645, 90, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(40)
        self.labelerode.setFont(font)
        self.labelerode.setTextFormat(QtCore.Qt.AutoText)
        self.labelerode.setObjectName("label")
        self.labelerode.setText("Erode:")

        self.sliderdialate = QtWidgets.QSlider(self.centralwidget)
        self.sliderdialate.setFocusPolicy(Qt.StrongFocus)
        self.sliderdialate.setGeometry(QtCore.QRect(470,680,100,50))
        self.sliderdialate.setOrientation(QtCore.Qt.Horizontal)
        self.sliderdialate.setTickPosition(QSlider.TicksBothSides)
        self.sliderdialate.setTickInterval(20)
        self.sliderdialate.setSingleStep(5)
        self.sliderdialate.valueChanged.connect(self.changeDialate)

        self.labeldialate = QtWidgets.QLabel(self.centralwidget)
        self.labeldialate.setGeometry(QtCore.QRect(470, 645, 90, 30))
        self.labeldialate.setFont(font)
        self.labeldialate.setTextFormat(QtCore.Qt.AutoText)
        self.labeldialate.setObjectName("label")
        self.labeldialate.setText("Dilate:")


        self.sliderkmeans = QtWidgets.QSlider(self.centralwidget)
        self.sliderkmeans.setFocusPolicy(Qt.StrongFocus)
        self.sliderkmeans.setGeometry(QtCore.QRect(600,680,100,50))
        self.sliderkmeans.setOrientation(QtCore.Qt.Horizontal)
        self.sliderkmeans.setTickPosition(QSlider.TicksBothSides)
        self.sliderkmeans.setTickInterval(20)
        self.sliderkmeans.setSingleStep(5)
        self.sliderkmeans.setValue(100)
        self.sliderkmeans.valueChanged.connect(self.changeK)

        self.labelK = QtWidgets.QLabel(self.centralwidget)
        self.labelK.setGeometry(QtCore.QRect(600, 645, 90, 30))
        self.labelK.setFont(font)
        self.labelK.setTextFormat(QtCore.Qt.AutoText)
        self.labelK.setObjectName("label")
        self.labelK.setText("K(Kmeans):")


        #----------------------------------------------------------------------------------------------

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 440, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.label.setText("Predicted:")

        self.text_label1 = QtWidgets.QLabel(self.centralwidget)
        self.text_label1.setGeometry(QtCore.QRect(170, 500, 151, 51))
        self.text_label1.setObjectName("text_label1")
        self.text_label1.setStyleSheet("border: 3px solid black;")
        self.text_label1.setAlignment(QtCore.Qt.AlignCenter)


        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(720, 440, 151, 51))
        self.label_2.setText("Sentence:")
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        
        self.text_label2 = QtWidgets.QLabel(self.centralwidget)
        self.text_label2.setGeometry(QtCore.QRect(720, 500, 151, 51))
        self.text_label2.setObjectName("text_label2")
        self.text_label2.setStyleSheet("border: 3px solid black;")
        self.text_label2.setAlignment(QtCore.Qt.AlignCenter)

        #----------------------------------------------------------------------------------------------
        self.upBtn = QtWidgets.QPushButton(self.centralwidget)
        self.upBtn.setGeometry(QtCore.QRect(510, 449, 25, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.upBtn.setFont(font)
        self.upBtn.setStyleSheet("color:rgb(85, 85, 255)")
        self.upBtn.setObjectName("pushButton")
        self.upBtn.setText("^")
        self.upBtn.clicked.connect(self.up)


        self.downBtn = QtWidgets.QPushButton(self.centralwidget)
        self.downBtn.setGeometry(QtCore.QRect(510, 538, 25, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.downBtn.setFont(font)
        self.downBtn.setStyleSheet("color:rgb(85, 85, 255)")
        self.downBtn.setObjectName("pushButton")
        self.downBtn.setText("v")
        self.downBtn.clicked.connect(self.down)

        self.leftBtn = QtWidgets.QPushButton(self.centralwidget)
        self.leftBtn.setGeometry(QtCore.QRect(450, 506, 50, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.leftBtn.setFont(font)
        self.leftBtn.setStyleSheet("color:rgb(85, 85, 255)")
        self.leftBtn.setObjectName("pushButton")
        self.leftBtn.setText("<")
        self.leftBtn.clicked.connect(self.left)

        self.rightBtn = QtWidgets.QPushButton(self.centralwidget)
        self.rightBtn.setGeometry(QtCore.QRect(544, 506, 50, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.rightBtn.setFont(font)
        self.rightBtn.setStyleSheet("color:rgb(85, 85, 255)")
        self.rightBtn.setObjectName("pushButton")
        self.rightBtn.setText(">")
        self.rightBtn.clicked.connect(self.right)


        self.autobtn = QtWidgets.QPushButton(self.centralwidget)
        self.autobtn.setGeometry(QtCore.QRect(510, 507, 25, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.autobtn.setFont(font)
        self.autobtn.setStyleSheet("color:rgb(85, 85, 255)")
        self.autobtn.setObjectName("pushButton")
        self.autobtn.setText("A")
        self.autobtn.clicked.connect(self.auto)
        #----------------------------------------------------------------------------------------------

        

        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setGeometry(QtCore.QRect(350, 600, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton1.setFont(font)
        self.pushButton1.setStyleSheet("color:rgb(85, 85, 255)")
        self.pushButton1.setObjectName("pushButton")
        self.pushButton1.setText("Clear")
        self.pushButton1.clicked.connect(self.clearButtonCliked)
        
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(550, 600, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton2.setFont(font)
        self.pushButton2.setStyleSheet("color:rgb(85, 85, 255)")
        self.pushButton2.setObjectName("pushButton")
        self.pushButton2.setText("Speak")
        self.pushButton2.clicked.connect(self.speechButtonCliked)

        self.changemodelLabel = QtWidgets.QPushButton(self.centralwidget)
        self.changemodelLabel.setGeometry(QtCore.QRect(150, 600, 150, 40))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.changemodelLabel.setFont(font)
        self.changemodelLabel.setStyleSheet("color:rgb(85, 85, 255)")
        self.changemodelLabel.setObjectName("pushButton")
        self.model="American"
        self.changemodelLabel.setText("American")
        self.changemodelLabel.clicked.connect(self.changeModelCliked)

        self.pushButton3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton3.setGeometry(QtCore.QRect(730, 600, 120, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton3.setFont(font)
        self.pushButton3.setStyleSheet("color:rgb(85, 85, 255)")
        self.pushButton3.setObjectName("pushButton")
        self.pushButton3.setText("<-")
        self.pushButton3.clicked.connect(self.clearOnceButtonCliked)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")

        self.thread1 = VideoThread(self.text_label1,self.text_label2)
        self.thread1.change_pixmap_signal1.connect(self.update_image1)
        self.thread1.change_pixmap_signal2.connect(self.update_image2)
        self.thread1.start()
        QtCore.QMetaObject.connectSlotsByName(self)
        thread=threading.Thread(target=self.speechButtonCliked)
        thread.start()
        thread.join()
    
    def changeErode(self,value):
        handDetect.erode(value//10)
    
    def changeDialate(self,value):
        handDetect.dialate(value//10)
    
    def changeK(self,value):
        handDetect.kmeans(value+2)
        

    def up(self,event):
        handDetect.move(u=True)
    def down(self,event):
        handDetect.move(d=True)
    def left(self,event):
        handDetect.move(l=True)
    def right(self,event):
        handDetect.move(r=True)
    def auto(self,event):
        handDetect.move(auto=True)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Hand Gesture Recognition', 'Are you sure you want to close the window?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply==QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    def clearButtonCliked(self,event):
        self.text_label2.clear()

    def changeModelCliked(self,event):
        if self.model=="American":
            self.thread1.changeModel(self.changemodelLabel.text())
            self.changemodelLabel.setText("Indian")
            self.model="Indian"
        else:
            self.thread1.changeModel(self.changemodelLabel.text())
            self.changemodelLabel.setText("American")
            self.model="American"
    
    def clearOnceButtonCliked(self,event):
        word=self.text_label2.text()
        self.text_label2.setText(word[:-1])

    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img,571,371)
        self.image_label1.setPixmap(qt_img)

    def update_image2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img,351,381)
        self.image_label2.setPixmap(qt_img)

    def speechButtonCliked(self,event):
        t1 = gtts.gTTS(self.text_label2.text())
        t1.save(os.path.join("temp","welcome.mp3")) 
        try: 
            pygame.mixer.init()
            pygame.mixer.music.load(os.path.join("temp","welcome.mp3"))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): # check if the file is playing
                pass
            pygame.mixer.quit()
        except:
            pass
        os.remove(os.path.join("temp","welcome.mp3"))
    
    def convert_cv_qt(self, cv_img,disply_width,display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
