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

class ModelWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Trainer") #sets title of the window
        self.setGeometry(200,200,700,500) #sets window to appear 600 pixels from the left and 600 from the top with a size of 300 x 300
        self.Vlayout = QVBoxLayout()
        self.setLayout(self.Vlayout) 
        self.box = QtWidgets.QFrame() 
        self.box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        progress_label = QtWidgets.QLabel("First, press 'TRAIN MODEL' to begin training!")
        self.box_text = QVBoxLayout()
        self.box_text.setAlignment(Qt.AlignTop)
        self.box_text.addWidget(progress_label)
        self.box.setLayout(self.box_text)
        self.bar = QProgressBar()
        self.bar.setMaximum(100)
        self.b2 = QPushButton("Train Model")
        self.b3 = QPushButton("Test Model")
        self.bMnist = QPushButton("MNIST Examples")
        self.b4 = QPushButton("Exit")
        self.button_layout = QHBoxLayout()
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
        self.train_thread = TrainThread()
        self.test_thread = TestThread()
        self.b2.clicked.connect(self.train_model)
        self.b3.clicked.connect(self.test_model)
        self.show()

    def train_model(self):
        self.b2.setEnabled(False)
        label_train = QtWidgets.QLabel("Model is being trained... see progress below.")
        self.box_text.addWidget(label_train)
        self.box.setLayout(self.box_text)
        self.train_thread.task_fin.connect(self.setValue)
        self.train_thread.start()
        self.train_thread.finished.connect(self.updateTrain)
        
    def setValue(self, value):
        self.bar.setValue(value)

    def updateTrain(self):
        label_train_cmp = QtWidgets.QLabel("Model training is complete. Press TEST MODEL for testing.")
        self.box_text.addWidget(label_train_cmp)
        self.box.setLayout(self.box_text)
        self.b3.setEnabled(True)
    
    def test_model(self):
        label_test = QtWidgets.QLabel("Model is being tested...")
        self.box_text.addWidget(label_test)
        self.box.setLayout(self.box_text)
        self.test_thread.task_fin.connect(self.setValue)
        self.test_thread.start()
        self.test_thread.finished.connect(self.updateTest)

    def updateTest(self):
        global Accuracy1
        label_test_cmp = QLabel("Model testing is complete. Accuracy: %0.2f%% " % Accuracy1)
        self.box_text.addWidget(label_test_cmp)
        self.box.setLayout(self.box_text)
        self.button_layout.itemAt(0).widget().deleteLater()
        self.button_layout.itemAt(1).widget().deleteLater()
        self.b5 = QPushButton("Drawing Canvas")
        self.b5.clicked.connect(self.drawingButton)
        self.button_layout.insertWidget(0,self.b5)

    def drawingButton(self):
        self.newCanvas = CanvasWindow()
        self.newCanvas.initUI()

    def viewExample(self):
        self.exampleWin = TrainImages()
        self.exampleWin.initExample()

    def clickExit(self):
        if self.train_thread.isRunning():
            self.train_thread.terminate()
        if self.test_thread.isRunning():
            self.test_thread.terminate()
        self.close()

class TrainThread(QThread):
    task_fin = pyqtSignal(int)

    def run(self):
        self.task_fin.emit(1)
        for epoch in range(1,11):
            train(epoch = epoch)
            time.sleep(0.3)
            self.task_fin.emit(epoch*10)

Accuracy1 = 0     
class TestThread(QThread):
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
        self.setGeometry(100,100,800,700)
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
    win = ModelWindow()
    win.show()
    sys.exit(app.exec_())

window()