# This script contains the code that creates a class which can take an input drawing of a digit via mouse. 
# It will present a canvas for the drawing. 
# This will be incorporated into the overall GUI of the system, and the DNN model will be used to predict the digit drawn.
# Last update: 9 April

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPainterPath, QPainter

class Canvas(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath() # class that helps with drawing operations

    # draw the path stored using QPainter class 
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPath(self.path)

    # When mouse is pressed, update path starting point
    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())
        self.update()

    # When mouse is pressed, update path starting point
    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        self.update()

    # Recommended widget size property
    def sizeHint(self): 
        return QSize(400, 400)

# For testing purposes
if __name__ == '__main__':
    app = QApplication(sys.argv)
    drawer = Canvas()
    drawer.show()
    sys.exit(app.exec_())

