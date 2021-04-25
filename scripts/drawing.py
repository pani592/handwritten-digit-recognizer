import sys
from pytorch import recognize
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFrame
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPainterPath, QPainter, QImage, QPen, QPixmap

import numpy as np
from PIL import Image, ImageQt, ImageOps

class Canvas(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)  # or instead use: super().__init__() 
        self.image = QImage(550, 600, QImage.Format_Grayscale8)   # initially did Format_RGB32 but MNIST is 8bit grayscale so this improves accuracy.
        self.blankCanvas()   # instantiate QPointerPath and sets white background
        self.penWidth = 30   # large pen width for better accuracy
        self.penColour = Qt.black

    # Creates a White canvas for drawing
    def blankCanvas(self):
        self.path = QPainterPath()
        self.image.fill(Qt.white)
        self.update()   # allows blankCanvas() to be used to clear canvas

    # Called whenever widget needs to be repainted
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())   

    # Called when a mouse button is pressed (inside the widget) 
    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    # Called when mouse moves with any mouse button (left or right or scroll) held down
    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        painter = QPainter(self.image)   
        painter.setPen(QPen(self.penColour, self.penWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))  # setting style of pen during motion
        painter.drawPath(self.path) # draw path
        self.update()   # Call paintEvent function

    # # Widget size
    def sizeHint(self): 
        return QSize(550, 600)

    def saveImage(self):
        image = ImageQt.fromqimage(self.image) # PIL to manipulate image before save
        image = ImageOps.invert(image)  # invert colour - MNIST is white on black bg.
        coords = image.getbbox()  # returns tuple of coords of the corners: left, top, right, bottom
        if coords != None: # if coords not empty
            coords = list(coords)
            coords[0] -= 50 # left pad 50
            coords[1] -= 10 # top pad 10
            coords[2] += 50 # right pad 50
            coords[3] += 10 # bottom pad 10
        image = image.crop(coords) # crop to boundary box
        image.save('digit.jpg') # save as jpg
        image = image.resize((20, 20), Image.ANTIALIAS) # resize to 20x20
        image.save('digit_inv_20x20.jpg') # save as jpg

        self.blankCanvas() # when image is saved, the canvas is cleared

#FIX ALL 'Self' sections to prevent warning error
class CanvasWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(600,200,1100,600) #sets window to appear 600 pixels from the left and 200 from the top with a size of 1100 x 550 
        self.Hlayout = QHBoxLayout()
        self.setLayout(self.Hlayout) #fix these 2 lines to prevent error from popping
        self.canvas = Canvas()
        self.predictBox = QtWidgets.QFrame() 
        self.predictBox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.numberSet = QHBoxLayout()
        for number in range(0,10):
            self.numberSet.addWidget(QtWidgets.QLabel("%d" %number))
        self.numberSet.setAlignment(Qt.AlignCenter)
        self.probabilityBox = QVBoxLayout()
        temp_label = QtWidgets.QLabel("Please draw a number and then press the recognize button.")
        self.prob_label = QtWidgets.QLabel("Hidden label", self)
        self.prob_label.clear()
        self.probabilityBox.addWidget(temp_label)
        self.probabilityBox.addLayout(self.numberSet)
        self.probabilityBox.addWidget(self.prob_label)
        self.probabilityBox.setAlignment(Qt.AlignTop)
        self.predictBox.setLayout(self.probabilityBox)
        self.clearButton = QPushButton("Clear Canvas")
        self.recogButton = QPushButton('Recognize Number')
        self.exit = QPushButton("Exit")
        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.clearButton)
        self.button_layout.addWidget(self.recogButton)
        self.button_layout.addWidget(self.predictBox)
        self.button_layout.addWidget(self.exit)
        self.Hlayout.addWidget(self.canvas)
        self.Hlayout.addLayout(self.button_layout)
        self.clearButton.clicked.connect(self.clear)
        self.recogButton.clicked.connect(self.recognizeButton)
        self.exit.clicked.connect(self.clickExit)

    def initUI(self):
        self.show()

    def clear(self):
        self.reset()
        self.canvas.blankCanvas()

    def recognizeButton(self):
        self.reset()
        self.canvas.saveImage()
        self.recog = recogThread()
        self.recog.start()
        self.recog.finished.connect(self.updatePredict)

    def updatePredict(self):
        global predicted_num
        pre_num = QtWidgets.QLabel("%d" % predicted_num)
        bigBold = pre_num.font()
        bigBold.setPointSize(20)
        bigBold.setBold(True)
        pre_num.setFont(bigBold)
        self.numberSet.itemAt(predicted_num).widget().deleteLater()
        self.numberSet.insertWidget(predicted_num,pre_num)
        prob_pix = QPixmap('class_prob.jpg')
        self.prob_label.setPixmap(prob_pix)

    def reset(self):
        self.prob_label.clear()
        for number in range(0,10):
            self.numberSet.itemAt(number).widget().deleteLater()
        for number in range(0,10):
            self.numberSet.insertWidget(number, QtWidgets.QLabel("%d" %number))

    def clickExit(self):
        self.close()

predicted_num = 9999

class recogThread(QThread):
    task_fin = pyqtSignal(int)

    def run(self):
        global predicted_num
        predicted_num, probab = recognize()

# For testing purposes
def drawingCanvas():
    app = QApplication(sys.argv)
    win = CanvasWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    ''' testing model - only runs when this script is run directly'''
    drawingCanvas()