# This script contains the code that creates a class which can take an input drawing of a digit via mouse. 
# It will present a canvas for the drawing. 
# This will be incorporated into the overall GUI of the system, and the DNN model will be used to predict the digit drawn.
# Last update: 18 April

import sys
from pytorch import recognize
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFrame
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPainterPath, QPainter, QImage, QPen

import numpy as np
from PIL import Image, ImageQt, ImageOps

class Canvas(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)  # or instead use: super().__init__() 
        self.image = QImage(400, 400, QImage.Format_Grayscale8)   # initially did Format_RGB32 but MNIST is 8bit grayscale so this improves accuracy.
        self.blankCanvas()   # instantiate QPointerPath and sets white background
        self.penWidth = 40   # large pen width for better accuracy
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
        return QSize(400, 400)

    # Change pen colour
    def newPenColour(self, colour):
        self.penColour = colour

    # Change pen width
    def newPenWidth(self, width):
        self.penWidth = width

    def saveImage(self):
        self.image.save('digit.jpg', 'jpg')   #  directly save qimage to jpg
        # or using PIL to manipulate image before save:
        image = ImageQt.fromqimage(self.image)
        image = image.convert('L')  # ensure grayscale
        image = ImageOps.invert(image)  # invert colour
        image.save('digit_inv.jpg')

        img_28x28 = image.resize((28, 28), Image.ANTIALIAS)
        img_28x28.save('digit_inv_28x28.jpg')

        self.blankCanvas() #when image is saved, the canvas is cleared

class FullWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,700,400) #sets window to appear 600 pixels from the left and 600 from the top with a size of 300 x 300
        self.Hlayout = QHBoxLayout(self)
        self.setLayout(self.Hlayout)
        self.canvas = Canvas()
        self.numbox = QtWidgets.QFrame(self) 
        self.numbox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.number = QVBoxLayout(self)
        self.number.setAlignment(Qt.AlignTop)
        self.numbox.setLayout(self.number)
        self.clearButton = QPushButton("Clear Canvas")
        self.captureButton = QPushButton('Save Image')
        self.recogButton = QPushButton('Recognize Number')
        self.exit = QPushButton("Exit")
        self.button_layout = QVBoxLayout(self)
        self.button_layout.addWidget(self.clearButton)
        self.button_layout.addWidget(self.captureButton)
        self.button_layout.addWidget(self.recogButton)
        self.button_layout.addWidget(self.numbox)
        self.button_layout.addWidget(self.exit)
        self.Hlayout.addWidget(self.canvas)
        self.Hlayout.addLayout(self.button_layout)
        self.clearButton.clicked.connect(self.clear)
        self.captureButton.clicked.connect(self.capture)
        self.recogButton.clicked.connect(self.recognizeButton)
        self.exit.clicked.connect(self.clickExit)

    def initUI(self):
        self.show()

    def clear(self):
        self.canvas.blankCanvas()

    def capture(self):
        self.canvas.saveImage()

    def recognizeButton(self):
        self.recog = recogThread()
        self.recog.start()
        self.recog.finished.connect(self.updatePredict)

    def updatePredict(self):
        global predicted_num
        pre_num = QtWidgets.QLabel("%d" % predicted_num)
        self.number.addWidget(pre_num)
        self.numbox.setLayout(self.number)

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
    win = FullWindow()
    win.show()
    sys.exit(app.exec_())