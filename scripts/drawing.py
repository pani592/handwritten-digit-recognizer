# This script contains the code that creates a class which can take an input drawing of a digit via mouse. 
# It will present a canvas for the drawing. 
# This will be incorporated into the overall GUI of the system, and the DNN model will be used to predict the digit drawn.
# Last update: 9 April

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainterPath, QPainter, QImage, QPen

class Canvas(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)  # or instead use: super().__init__() 
        self.image = QImage(400, 400, QImage.Format_RGB32)
        self.blankCanvas()   # instantiate QPointerPath and sets white background
        self.penWidth = 10   
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

# For testing purposes
if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = QWidget()   # new widget to hold clear canvas button and potentially other buttons
    widget.setLayout(QVBoxLayout())

    canvas = Canvas()
    clearButton = QPushButton("Clear Canvas")

    widget.layout().addWidget(canvas)
    widget.layout().addWidget(clearButton)

    clearButton.clicked.connect(canvas.blankCanvas)

    # canvas.newPenColour(Qt.blue)  # choose pen colour
    # canvas.newPenWidth(4)         # choose width of pen
    widget.show()
    sys.exit(app.exec_())

