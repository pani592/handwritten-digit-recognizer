# This file Contains Class definitions and functions for instantiating the GUI for the software system. 
# Authors: Paulse Anithottam, Sidharth Varma
# Last updated: 28 April

# imports
from model import *
import sys
import numpy as np
from PIL import Image, ImageQt, ImageOps
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QProgressBar, QLineEdit, QHBoxLayout, QFrame, QMenuBar, QComboBox
from PyQt5.QtGui import QPainterPath, QPainter, QImage, QPen, QPixmap

class Canvas(QWidget):
    '''Canvas is the Drawing canvas implementation. Also performs some primary image manipulation and saving image'''
    def __init__(self, parent=None):
        super().__init__() 
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

    # image operations when saving image from the canvas onto file
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
        image = image.resize((20, 20), Image.ANTIALIAS) # resize to 20x20 with a high-quality downsampling filter
        image.save('digit_inv_20x20.jpg') # save as jpg
        self.blankCanvas() # when image is saved, the canvas is cleared

class CanvasWindow(QWidget):
    ''' CanvasWindow is the Window on which the canvas is placed. Adds features such as recognise button for displaying the image and prediction.'''
    def __init__(self):
        super().__init__()
        self.font = QFont()
        self.font.setPointSize(15)
        self.setGeometry(600,200,1100,600) #sets window to appear 600 pixels from the left and 200 from the top with a size of 1100 x 600 
        self.Hlayout = QHBoxLayout()
        self.setLayout(self.Hlayout) 
        self.canvas = Canvas()
        self.predictBox = QtWidgets.QFrame() 
        self.predictBox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.numberSet = QHBoxLayout()
        for number in range(0,10):
            self.numberSet.addWidget(QtWidgets.QLabel("%d" %number))  # displays all 10 digits in a line for better visual effect
            QtWidgets.QLabel("%d" %number).setFont(self.font)
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

class MNISTImages(QWidget):
    ''' MNISTImages is the window for displaying MNIST training images, with a button to display the next set'''
    def __init__(self):
        super().__init__()
        self.setGeometry(100,100,800,700)
        mnistLayout = QVBoxLayout()

        # Explanation about MNIST for the user
        text = "Here, you can view examples of what  goes into the Neural Network as it trains. \
            \nThe dataset being used to train is the MNIST dataset, which consists of 60,000 training images and 10,000 testing images. \
            \nThese images have been normalised and scaled to a 28x28 size image, and is passed into the Neural Network. \
            \nPressing the 'Next Set' button shows you another set of 35 random images from the training set, so you can get an idea of the images."
        self.label1 = QLabel(text)
        self.label1.setWordWrap(True)
        mnistLayout.addWidget(self.label1)

        buttons_mnist = QHBoxLayout()
        self.exampleLab = QtWidgets.QLabel(self)
        examplesImage = QPixmap('mnist_examples.jpg')  # this is the jpg that is displayed on the window
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
        show_MNIST_examples() # when this function is called, a new set of 64 examples from MNIST dataset is saved to files
        examplesImage = QPixmap('mnist_examples.jpg')
        self.exampleLab.setPixmap(examplesImage)

    def exitButton(self):
        self.close()

model_choice = 1
class MainWindow(QWidget):
    ''' MainWindow is the primary screen/window displayed. Contains buttons to train dataset, test dataset, 
        and open separate windows to show MNIST examples, and access the drawing canvas window '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandRite Home") 
        self.setGeometry(200,200,625,400) 
        self.Vlayout = QVBoxLayout()
        self.setLayout(self.Vlayout)

        # style settings
        self.box = QtWidgets.QFrame() 
        self.box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.font = QFont()
        self.font.setPointSize(15) 

        # add combobox to allow user to select between different models
        self.cb = QComboBox() 
        self.cb.addItems(["Model 1", "Model 2"])
        self.cb.currentIndexChanged.connect(self.select_model)
        self.Vlayout.addWidget(self.cb)   

        # Text box to communicate output
        self.box_text = QVBoxLayout()
        self.box_text.setAlignment(Qt.AlignTop)
        progress_label = QtWidgets.QLabel("Select a model above, and press TRAIN MODEL!")
        progress_label.setFont(self.font)
        self.box_text.addWidget(progress_label)
        self.box.setLayout(self.box_text)

        # progress bar and buttons
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
    
        # display the logo of the team
        logo = QtWidgets.QLabel()
        logo.setStyleSheet("QWidget {background-image: url(logo1.jpg)}")
        self.Vlayout.addWidget(logo)

        # define what function each button connects to
        self.b4.clicked.connect(self.clickExit)
        self.bMnist.clicked.connect(self.viewExample)
        self.train_thread = TrainThread()
        self.test_thread = TestThread()
        self.b2.clicked.connect(self.train_model)
        self.b3.clicked.connect(self.test_model)
        self.show()

    # connected to the combox which updates what model is being used
    def select_model(self,i):
        global model_choice
        model_choice = i+1
    
    # connected to the train model button, calls train() function from model.py
    def train_model(self):
        self.b2.setEnabled(False)
        label_train = QtWidgets.QLabel("Training the Model... meanwhile, see examples of MNIST below.")
        label_train.setFont(self.font)
        self.box_text.addWidget(label_train)
        self.box.setLayout(self.box_text)
        self.train_thread.task_fin.connect(self.setValue)
        self.train_thread.start()
        self.train_thread.finished.connect(self.updateTrain)
        
    def setValue(self, value):
        self.bar.setValue(value)

    # once training is complete, this function performs updates to the GUI like disabling the TRAIN button
    def updateTrain(self):
        label_train_cmp = QtWidgets.QLabel("Model training complete! Press TEST MODEL to see results.")
        label_train_cmp.setFont(self.font)
        self.box_text.addWidget(label_train_cmp)
        self.box.setLayout(self.box_text)
        self.b3.setEnabled(True)
    
    # Connected to the train model button
    def test_model(self):
        label_test = QtWidgets.QLabel("Testing the Model...")
        label_test.setFont(self.font)
        self.box_text.addWidget(label_test)
        self.box.setLayout(self.box_text)
        self.test_thread.task_fin.connect(self.setValue)
        self.test_thread.start()
        self.test_thread.finished.connect(self.updateTest)

    # updates made to the GUI once testing is complete
    def updateTest(self):
        global Accuracy1
        label_test_cmp = QLabel("Testing complete. Model Accuracy: %0.2f%% " % Accuracy1)
        label_test_cmp.setFont(self.font)
        self.box_text.addWidget(label_test_cmp)
        self.box.setLayout(self.box_text)
        self.button_layout.itemAt(0).widget().deleteLater()
        self.button_layout.itemAt(1).widget().deleteLater()
        self.b5 = QPushButton("Drawing Canvas")
        self.b5.clicked.connect(self.drawingButton)
        self.button_layout.insertWidget(0,self.b5)

    # new button to link to the next window, the drawing canvas
    def drawingButton(self):
        self.newCanvas = CanvasWindow() # initiates drawing canvas window class instance
        self.newCanvas.initUI()  # displays new window with drawing canvas and prediction screen

    # connected to the view MNIST examples button
    def viewExample(self):
        self.exampleWin = MNISTImages()
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
        global model_choice
        self.task_fin.emit(1)
        for epoch in range(1,2):  # 20 epochs
            train(epoch = epoch, input = model_choice)
            time.sleep(0.3)
            self.task_fin.emit(epoch*5) # progress bar updates in steps of 5%

Accuracy1 = 0     
class TestThread(QThread):
    task_fin = pyqtSignal(int)
    def run(self):
        global model_choice
        global Accuracy1
        tmp_acc = test(input = model_choice)
        Accuracy1 = tmp_acc*100
        time.sleep(0.2)
        self.task_fin.emit(100)

predicted_num = 9999
class recogThread(QThread):
    task_fin = pyqtSignal(int)

    def run(self):
        global predicted_num
        global model_choice
        predicted_num, probab = recognize(input=model_choice)

def view_UI():
    ''' The main function that is called to instantiate the window and start the system'''
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

def drawing_canvas():
    ''' When this function is called, the drawing window alone is generated. Useful for testing.'''
    app = QApplication(sys.argv)
    win = CanvasWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # view_UI()
    drawing_canvas()