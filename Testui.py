from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import handDetect


class VideoThread(QThread):
    change_pixmap_signal1 = pyqtSignal(np.ndarray)
    change_pixmap_signal2 = pyqtSignal(np.ndarray)

    def __init__(self,tl1,tl2):
        super().__init__()
        self._run_flag = True
        self.tl1=tl1
        self.tl2=tl2

    def run(self):
        while self._run_flag:
            handDetect.fun(self.change_pixmap_signal1,self.change_pixmap_signal2,self._run_flag,self.tl1,self.tl2)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.resize(1000,900)
        self.disply_width = 512
        self.display_height = 512
        # create the label that holds the image
        self.image_label1 = QLabel(self)
        self.image_label1.resize(self.disply_width, self.display_height)

        self.image_label2 = QLabel(self)
        self.image_label2.resize(self.disply_width, self.display_height)
        
        # create a text label
        self.textLabel1 = QLabel('')
        self.textLabel2 = QLabel('')

        # create a vertical box layout and add the two labels
        Gbox = QGridLayout()
        Gbox.addWidget(self.image_label1, 0, 0)
        Gbox.addWidget(self.image_label2, 0, 1)

        Gbox.addWidget(self.textLabel1,1,0)
        Gbox.addWidget(self.textLabel2,1,1)

        
        # set the vbox layout as the widgets layout
        
        self.setLayout(Gbox)
        

        # create the video capture thread
        self.thread1 = VideoThread(self.textLabel1,self.textLabel2)
        # connect its signal to the update_image slot
        self.thread1.change_pixmap_signal1.connect(self.update_image1)
        self.thread1.change_pixmap_signal2.connect(self.update_image2)
    
        self.thread1.start()

    def closeEvent(self, event):
        self.thread1.stop()
        event.accept()
        self.close()
        



    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label1.setPixmap(qt_img)

    def update_image2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label2.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())