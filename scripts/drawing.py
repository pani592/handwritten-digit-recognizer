# This script contains the code that creates a class which can take an input drawing of a digit via mouse. 
# It will present a canvas for the drawing. 
# This will be incorporated into the overall GUI of the system, and the DNN model will be used to predict the digit drawn.
# Last update: 9 April

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QSize, Qt
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

    # Widget size
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

        image_array = np.array(img_28x28) # 28 x 28 
        image_array = (image_array.flatten())  # array len 784
        # image_array  = image_array.reshape(-1,1).T  # shape is 1 X 784 now, not sure if reqd.
        image_array = image_array.astype('float32')
        image_array /= 255 
        # this image array will be input into the machine learning model.

# For testing purposes
if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = QWidget()   # new widget to hold clear canvas button and potentially other buttons
    widget.setLayout(QVBoxLayout())

    canvas = Canvas()
    clearButton = QPushButton("Clear Canvas")
    captureButton = QPushButton('Save Image')

    widget.layout().addWidget(canvas)
    widget.layout().addWidget(clearButton)
    widget.layout().addWidget(captureButton)

    clearButton.clicked.connect(canvas.blankCanvas)
    captureButton.clicked.connect(lambda: canvas.saveImage()) 

    # canvas.newPenColour(Qt.blue)  # choose pen colour
    # canvas.newPenWidth(4)         # choose width of pen
    widget.show()
    sys.exit(app.exec_())