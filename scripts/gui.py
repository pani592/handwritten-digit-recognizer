#script for gui interface

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import sys

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.initUI()
        self.setGeometry(500,500,600,600) #sets window to appear 500 pixels from the left and 500 from the top with a size of 600 x 600
        self.setWindowTitle("Handwritten Digit Recognizer")
    
    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("No press")
        self.label.move(100,100)
        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setGeometry(250,200, 100, 40)
        self.button1.setText("Select Model")
        self.button1.clicked.connect(self.clickedSelect)
        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setGeometry(250, 250, 100, 40)
        self.button2.setText("Exit")
        self.button2.clicked.connect(self.clickExit)

    def clickedSelect(self):
        self.label.setText("You pressed the button ;)")
        self.update()

    def clickExit(self):
        self.close()

    def update(self):
        self.label.adjustSize()

#class SecondWin(QWidget):




def window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

window()