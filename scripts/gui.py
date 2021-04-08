#script for gui interface

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout
import sys

class UI(QWidget):
    #best implementation in terms of dynamically changing window and creating new options
    #Dynamically adjusts the window and buttons, adds new ones when one is pressed
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setGeometry(500,500,300,300) #sets window to appear 500 pixels from the left and 500 from the top with a size of 600 x 600
        self.setWindowTitle("Handwritten Digit Recognizer")

    def initUI(self):
        self.ModelSel = QPushButton("Select Model")
        self.Exit = QPushButton("Exit")
        self.ModelSel.clicked.connect(self.ModelOps) #connects button press to function "ModelOps"
        self.Exit.clicked.connect(self.clickExit) #connects to "clickExit"

        self.layout = QGridLayout() #Creates a grid layout where widgets can be added into a particular positon easily
        # allows for automatic dynamic adjusting
        self.layout.addWidget(self.ModelSel, 0 , 0) #adds Model Selection button at (0,0) position
        self.layout.addWidget(self.Exit, 3, 0) #sets Exit to be on the 3rd row but does not have a have a gap until button1 and button2 are added in rows 1 and 2

        self.setLayout(self.layout)
        self.show()

    def ModelOps(self):
        button1 = QPushButton("Model 1")
        button2 = QPushButton("Model 2")
        self.layout.addWidget(button1, 1, 0)
        self.layout.addWidget(button2, 2, 0)
        button1.clicked.connect(self.Model1)
        button2.clicked.connect(self.Model2)
        self.layout.itemAt(0).widget().deleteLater()

    def Model1(self):
        self.NewWin1 = SecondWin()
        self.NewWin1.initModel1()
    
    def Model2(self):
        self.NewWin2 = SecondWin()
        self.NewWin2.initModel2()

    def clickExit(self):
        self.close()

# Initial implementation of UI with a basic starting page
# class Window(QMainWindow):
#     def __init__(self):
#         super(Window, self).__init__()
#         self.initUI()
#         self.setGeometry(500,500,600,600) #sets window to appear 500 pixels from the left and 500 from the top with a size of 600 x 600
#         self.setWindowTitle("Handwritten Digit Recognizer")
    
#     def initUI(self):
#         self.label = QtWidgets.QLabel(self)
#         self.label.setText("No press")
#         self.label.move(100,100)
#         self.button1 = QtWidgets.QPushButton(self)
#         self.button1.setGeometry(250,200, 100, 40)
#         self.button1.setText("Select Model")
#         self.button1.clicked.connect(self.clickedSelect)
#         self.button2 = QtWidgets.QPushButton(self)
#         self.button2.setGeometry(250, 250, 100, 40)
#         self.button2.setText("Exit")
#         self.button2.clicked.connect(self.clickExit)

#     def clickedSelect(self):
#         self.button1.deleteLater()
#         self.label = QtWidgets.QLabel(self)
#         self.label.setText("Please select a model from one of the below:")
#         self.update()
#         self.label.move(100,100)
#         self.model1 = QtWidgets.QPushButton(self)
#         self.model1.setGeometry(150,100, 200, 40)
#         self.model1.setText("Model 1")
#         self.model2 = QtWidgets.QPushButton(self)
#         self.model2.setGeometry(150,150, 200, 40)
#         self.model2.setText("Model 2")
#         self.model3 = QtWidgets.QPushButton(self)
#         self.model3.setGeometry(150,200, 200, 40)
#         self.model3.setText("Model 3")
        
#         self.update()
        
#         self.New = SecondWin()
#         self.New.show()
#         w = SecondWin()
#         w.show()
#         self.close()

#     def clickExit(self):
#         self.close()

#     def update(self):
#         self.label.adjustSize()

#Originally meant for second window to give model selection options, will be repurposed to show model training progress
class SecondWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(600,600,300,300) #sets window to appear 500 pixels from the left and 500 from the top with a size of 600 x 600
        self.setWindowTitle("Select Handwriting Recognition Model") #sets title of the window

    def initModel1(self):
        '''
        Used for when model 1 is selected
        '''
        self.label = QtWidgets.QLabel(self)
        self.label.setText("You have selected Model 1")
        self.label.move(80,80)
        self.update()
        self.show()

    def initModel2(self):
        '''
        Used for when model 2 is selected
        '''
        self.label = QtWidgets.QLabel(self)
        self.label.setText("You have selected Model 2")
        self.label.move(80,80)
        self.update()
        self.show()
    
    def update(self):
        '''
        Updates the label to adjust based on the current text 
        Prevents text from being cropped out
        '''
        self.label.adjustSize()





def window():
    app = QApplication(sys.argv)
    win = UI()
    win.show()
    sys.exit(app.exec_())

window()