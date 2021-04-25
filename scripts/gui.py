#script for gui interface

from pytorch import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QProgressBar, QLineEdit, QHBoxLayout, QFrame, QMenuBar
from drawing import *
import sys
import time
import torch

class UI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setGeometry(150,150,300,300) #sets window to appear 500 pixels from the left and 500 from the top with a size of 600 x 600
        self.setWindowTitle("Handwritten Digit Recognizer")

    def initUI(self):
        self.ModelSel = QPushButton("Select Model")
        self.Exit = QPushButton("Exit")
        self.ModelSel.clicked.connect(self.ModelOps) #connects button press to function "ModelOps"
        self.Exit.clicked.connect(self.clickExit) #connects to "clickExit"

        self.layout = QGridLayout() #Creates a grid layout where widgets can be added into a particular positon easily
        # allows for automatic dynamic changes
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
        self.NewWin1 = Model1()
        self.NewWin1.initModel()
    
    def Model2(self):
        self.NewWin2 = Model1()
        self.NewWin2.initModel()

    def clickExit(self):
        self.close()


box_text = QVBoxLayout()
box_text.setAlignment(Qt.AlignTop)

class Model1(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,500,300) #sets window to appear 600 pixels from the left and 600 from the top with a size of 300 x 300
        self.Vlayout = QVBoxLayout(self)
        self.setLayout(self.Vlayout) 
        self.box = QtWidgets.QFrame(self) 
        self.box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        progress_label = QtWidgets.QLabel("First, press 'TRAIN MODEL' to begin training!")
        box_text.addWidget(progress_label)
        self.box.setLayout(box_text)
        self.bar = QProgressBar(self)
        self.bar.setMaximum(100)
        self.b2 = QPushButton("Train Model")
        self.b3 = QPushButton("Test Model")
        self.bMnist = QPushButton("MNIST Examples")
        self.b4 = QPushButton("Exit")
        self.button_layout = QHBoxLayout(self)
        self.button_layout.addWidget(self.b2)
        self.button_layout.addWidget(self.b3)
        self.button_layout.addWidget(self.bMnist)
        self.button_layout.addWidget(self.b4)
        self.b3.setEnabled(False)
        self.Vlayout.addWidget(self.box)
        self.Vlayout.addWidget(self.bar)
        self.Vlayout.addLayout(self.button_layout)
        self.b4.clicked.connect(self.clickExit)
        
        self.bMnist.clicked.connect(self.viewExample)

    def initModel(self):
        '''
        Used for when model 1 is selected
        '''
        self.setWindowTitle("Model 1") #sets title of the window
        self.train_thread1 = TrainThread1()
        self.test_thread1 = TestThread1()
        self.b2.clicked.connect(self.train_model1)
        self.b3.clicked.connect(self.test_model1)
        self.show()

    def train_model1(self):
        self.b2.setEnabled(False)
        label_train = QtWidgets.QLabel("Model is being trained... see progress below.")
        box_text.addWidget(label_train)
        self.box.setLayout(box_text)
        self.train_thread1.task_fin.connect(self.setValue)
        self.train_thread1.start()
        self.train_thread1.finished.connect(self.updateTrain1)
        

    def setValue(self, value):
        self.bar.setValue(value)

    def updateTrain1(self):
        label_train_cmp = QtWidgets.QLabel("Model training is complete. Press TEST MODEL for testing.")
        box_text.addWidget(label_train_cmp)
        self.box.setLayout(box_text)
        self.b3.setEnabled(True)
    
    def test_model1(self):
        label_test = QtWidgets.QLabel("Model is being tested...")
        box_text.addWidget(label_test)
        self.box.setLayout(box_text)
        self.test_thread1.task_fin.connect(self.setValue)
        self.test_thread1.start()
        self.test_thread1.finished.connect(self.updateTest1)

    def updateTest1(self):
        global Accuracy1
        label_test_cmp = QLabel("Model testing is complete. Accuracy: %0.2f%% " % Accuracy1)
        box_text.addWidget(label_test_cmp)
        self.box.setLayout(box_text)
        self.button_layout.itemAt(0).widget().deleteLater()
        self.button_layout.itemAt(1).widget().deleteLater()
        self.b5 = QPushButton("Drawing Canvas")
        self.b5.clicked.connect(self.drawingButton)
        self.button_layout.insertWidget(0,self.b5)

    def drawingButton(self):
        self.newCanvas = CanvasWindow()
        self.newCanvas.initUI()

    def initModel2(self):
        '''
        Used for when model 2 is selected
        '''
        self.setWindowTitle("Model 2") #sets title of the window
        self.show()

    def clearLayout(self):
        for i in reversed(range(box_text.count())): 
            print(i)
            box_text.itemAt(i).widget().deleteLater()

    def viewExample(self):
        self.exampleWin = TrainImages()
        self.exampleWin.initExample()

    def clickExit(self):
        self.clearLayout()
        if self.train_thread1.isRunning():
            self.train_thread1.terminate()
        if self.test_thread1.isRunning():
            self.test_thread1.terminate()
        self.close()

class Model2(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200,500,300) #sets window to appear 600 pixels from the left and 600 from the top with a size of 300 x 300
        self.Vlayout = QVBoxLayout(self)
        self.setLayout(self.Vlayout) 
        self.box = QtWidgets.QFrame(self) 
        self.box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        progress_label = QtWidgets.QLabel("First, press 'TRAIN MODEL' to begin training!")
        box_text.addWidget(progress_label)
        self.box.setLayout(box_text)
        self.bar = QProgressBar(self)
        self.bar.setMaximum(100)
        self.b2 = QPushButton("Train Model")
        self.b3 = QPushButton("Test Model")
        self.bMnist = QPushButton("MNIST Examples")
        self.b4 = QPushButton("Exit")
        self.button_layout = QHBoxLayout(self)
        self.button_layout.addWidget(self.b2)
        self.button_layout.addWidget(self.b3)
        self.button_layout.addWidget(self.bMnist)
        self.button_layout.addWidget(self.b4)
        self.b3.setEnabled(False)
        self.Vlayout.addWidget(self.box)
        self.Vlayout.addWidget(self.bar)
        self.Vlayout.addLayout(self.button_layout)
        self.b4.clicked.connect(self.clickExit)
        
        self.bMnist.clicked.connect(self.viewExample)

    def initModel(self):
        '''
        Used for when model 1 is selected
        '''
        self.setWindowTitle("Model 1") #sets title of the window
        self.train_thread1 = TrainThread1()
        self.test_thread1 = TestThread1()
        self.b2.clicked.connect(self.train_model1)
        self.b3.clicked.connect(self.test_model1)
        self.show()

    def train_model1(self):
        self.b2.setEnabled(False)
        label_train = QtWidgets.QLabel("Model is being trained... see progress below.")
        box_text.addWidget(label_train)
        self.box.setLayout(box_text)
        self.train_thread1.task_fin.connect(self.setValue)
        self.train_thread1.start()
        self.train_thread1.finished.connect(self.updateTrain1)
        

    def setValue(self, value):
        self.bar.setValue(value)

    def updateTrain1(self):
        label_train_cmp = QtWidgets.QLabel("Model training is complete. Press TEST MODEL for testing.")
        box_text.addWidget(label_train_cmp)
        self.box.setLayout(box_text)
        self.b3.setEnabled(True)
    
    def test_model1(self):
        label_test = QtWidgets.QLabel("Model is being tested...")
        box_text.addWidget(label_test)
        self.box.setLayout(box_text)
        self.test_thread1.task_fin.connect(self.setValue)
        self.test_thread1.start()
        self.test_thread1.finished.connect(self.updateTest1)

    def updateTest1(self):
        global Accuracy1
        label_test_cmp = QLabel("Model testing is complete. Accuracy: %0.2f%% " % Accuracy1)
        box_text.addWidget(label_test_cmp)
        self.box.setLayout(box_text)
        self.button_layout.itemAt(0).widget().deleteLater()
        self.button_layout.itemAt(1).widget().deleteLater()
        self.b5 = QPushButton("Drawing Canvas")
        self.b5.clicked.connect(self.drawingButton)
        self.button_layout.insertWidget(0,self.b5)

    def drawingButton(self):
        self.newCanvas = CanvasWindow()
        self.newCanvas.initUI()

    def initModel2(self):
        '''
        Used for when model 2 is selected
        '''
        self.setWindowTitle("Model 2") #sets title of the window
        self.show()

    def clearLayout(self):
        for i in reversed(range(box_text.count())): 
            print(i)
            box_text.itemAt(i).widget().deleteLater()

    def viewExample(self):
        self.exampleWin = TrainImages()
        self.exampleWin.initExample()

    def clickExit(self):
        self.clearLayout()
        if self.train_thread1.isRunning():
            self.train_thread1.terminate()
        if self.test_thread1.isRunning():
            self.test_thread1.terminate()
        self.close()

class TrainThread1(QThread):
    task_fin = pyqtSignal(int)

    def run(self):
        count = 0
        self.task_fin.emit(2)
        for epoch in range(1,11):
            count += 1
            train(epoch)
            time.sleep(0.3)
            self.task_fin.emit(count*10)
        torch.save(model, './pytorch_model.pth')


Accuracy1 = 0     
class TestThread1(QThread):
    task_fin = pyqtSignal(int)

    def run(self):
        global Accuracy1
        tmp_acc = test()
        Accuracy1 = tmp_acc*100
        time.sleep(0.2)
        self.task_fin.emit(100)

class TrainImages(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100,100,800,600)
        mnistLayout = QVBoxLayout()
        buttons_mnist = QHBoxLayout()
        self.exampleLab = QtWidgets.QLabel(self)
        examplesImage = QPixmap('mnist_examples.jpg')
        self.exampleLab.setPixmap(examplesImage)
        sizeP = self.exampleLab.sizePolicy()
        sizeP.setHorizontalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.exampleLab.setSizePolicy(sizeP)
        mnistLayout.addWidget(self.exampleLab)
        self.nextBut = QPushButton("Next Set", self)
        self.exitBut = QPushButton("Exit", self)
        self.nextBut.clicked.connect(self.nextRand)
        self.exitBut.clicked.connect(self.exitButton)
        buttons_mnist.addWidget(self.nextBut)
        buttons_mnist.addWidget(self.exitBut)
        self.layout = QVBoxLayout()
        # self.layout.addWidget(self.exitBut)
        self.layout.addLayout(mnistLayout)
        self.layout.addLayout(buttons_mnist)
        self.setLayout(self.layout)
    
    def initExample(self):
        self.show()

    def nextRand(self):
        showMNISTExamples()
        examplesImage = QPixmap('mnist_examples.jpg')
        self.exampleLab.setPixmap(examplesImage)

    def exitButton(self):
        self.close()



def window():
    app = QApplication(sys.argv)
    win = UI()
    win.show()
    sys.exit(app.exec_())

window()