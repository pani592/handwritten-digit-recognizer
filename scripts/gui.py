# This file Contains Class definitions and functions for instantiating the GUI for the software system. 
# Authors: Paulse Anithottam, Sidharth Varma
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
        # created font to use on the numbers displayed
        self.font = QFont()
        self.font.setPointSize(15)
        self.setGeometry(600,200,1100,600) #sets window to appear 600 pixels from the left and 200 from the top with a size of 1100 x 600 

        # setting layout for current widget
        self.Hlayout = QHBoxLayout()
        self.setLayout(self.Hlayout)

        # setting up a Canvas widget class and box that changes text after button presses
        self.canvas = Canvas() 
        self.predictBox = QtWidgets.QFrame() 
        self.predictBox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        # set up for horizontal line of numbers
        self.numberSet = QHBoxLayout()
        for number in range(0,10):
            self.numberSet.addWidget(QtWidgets.QLabel("%d" %number))  # displays all 10 digits in a line for better visual effect
            QtWidgets.QLabel("%d" %number).setFont(self.font) #setting font for each
        self.numberSet.setAlignment(Qt.AlignCenter) # aligns the entire layout center

        # adding widgets together in VBoxLayout
        self.probabilityBox = QVBoxLayout()
        temp_label = QtWidgets.QLabel("Please draw a number and then press the recognize button.")
        # set up temporary label where probability graph is added later
        self.prob_label = QtWidgets.QLabel("Hidden label", self)
        self.prob_label.clear()
        # adding the components to another sub layout 
        self.probabilityBox.addWidget(temp_label)
        self.probabilityBox.addLayout(self.numberSet)
        self.probabilityBox.addWidget(self.prob_label)
        self.probabilityBox.setAlignment(Qt.AlignTop)
        self.predictBox.setLayout(self.probabilityBox)

        # adding buttons and connecting them to their functionality
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
        # shows the UI
        self.show()

    def clear(self):
        # clears the canvas and resets the text 
        self.reset()
        self.canvas.blankCanvas()

    def recognizeButton(self):
        # resets text
        self.reset()
        # saves drawing and pushes it into the recognise QThread
        self.canvas.saveImage()
        self.recog = recogThread()
        self.recog.start()
        # updates text after a number is recognised
        self.recog.finished.connect(self.updatePredict)

    def updatePredict(self):
        # bolds text and increases text size for number recognised by model
        global predicted_num
        pre_num = QtWidgets.QLabel("%d" % predicted_num)
        bigBold = pre_num.font()
        bigBold.setPointSize(20)
        bigBold.setBold(True)
        pre_num.setFont(bigBold)
        self.numberSet.itemAt(predicted_num).widget().deleteLater()
        self.numberSet.insertWidget(predicted_num,pre_num)
        # adds the class probability image made in the view_probabilities() function
        prob_pix = QPixmap('class_prob.jpg')
        self.prob_label.setPixmap(prob_pix)

    def reset(self):
        # clears class probability 
        self.prob_label.clear()
        # deletes all the numbers in the numberSet layout to remove any bold text
        for number in range(0,10):
            self.numberSet.itemAt(number).widget().deleteLater()
        # adds all the numbers back in regular text
        for number in range(0,10):
            self.numberSet.insertWidget(number, QtWidgets.QLabel("%d" %number))

    def clickExit(self):
        # closes the drawing canvas widget
        self.close()

class MNISTImages(QWidget):
    ''' MNISTImages is the window for displaying MNIST training images, with a button to display the next set'''
    def __init__(self):
        super().__init__()
        # setting geometry
        self.setGeometry(100,100,800,700)
        # setting layout to add widgets and font size for text
        mnistLayout = QVBoxLayout()
        self.font = QFont()
        self.font.setPointSize(13)

        # Explanation about MNIST for the user
        text = "Here, you can view examples of what  goes into the Neural Network as it trains. " + \
            "The dataset being used to train is the MNIST dataset, which consists of 60,000 training images and 10,000 testing images. " + \
            "These images have been normalised and scaled to a 28x28 size image, and is passed into the Neural Network. " + \
            "Pressing the 'Next Set' button shows you another set of 35 random images from the training set, so you can get an idea of the images. " + \
            "Please wait 1/2 seconds after pressing 'Next Set' for it to load. "
        self.label1 = QLabel(text)
        self.label1.setWordWrap(True)
        self.label1.setFont(self.font)
        mnistLayout.addWidget(self.label1)

        # layout for buttons 
        buttons_mnist = QHBoxLayout()
        # adding image using QPixmap to QLabel
        self.exampleLab = QtWidgets.QLabel(self)
        examplesImage = QPixmap('mnist_examples.jpg')  # this is the jpg that is displayed on the window
        self.exampleLab.setPixmap(examplesImage)
        sizeP = self.exampleLab.sizePolicy() 
        sizeP.setHorizontalPolicy(QtWidgets.QSizePolicy.Maximum) # sets horizontal size policy for mnist image
        self.exampleLab.setSizePolicy(sizeP)
        mnistLayout.addWidget(self.exampleLab)
        # creating and adding buttons to layout
        self.nextBut = QPushButton("Next Set", self)
        self.exitBut = QPushButton("Exit", self)
        # button functionality
        self.nextBut.clicked.connect(self.nextRand)
        self.exitBut.clicked.connect(self.exitButton)
        buttons_mnist.addWidget(self.nextBut)
        buttons_mnist.addWidget(self.exitBut)
        self.layout = QVBoxLayout()
        self.layout.addLayout(mnistLayout)
        self.layout.addLayout(buttons_mnist)
        self.setLayout(self.layout) # sets widget layout
    
    def initExample(self):
        # shows UI
        self.show()

    def nextRand(self):
        # connected to "Next Set" button
        show_MNIST_examples() # when this function is called, a new set of 64 examples from MNIST dataset is saved to files
        examplesImage = QPixmap('mnist_examples.jpg')
        self.exampleLab.setPixmap(examplesImage) # replaces old set with the new set of MNIST examples

    def exitButton(self):
        # closes UI window
        self.close()

# global variable changed according to combobox selection
model_choice = 1
class MainWindow(QWidget):
    ''' MainWindow is the primary screen/window displayed. Contains buttons to train dataset, test dataset, 
        and open separate windows to show MNIST examples, and access the drawing canvas window '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandRite Home") 
        self.setGeometry(200,200,625,400) 
        # initialises layout
        self.Vlayout = QVBoxLayout()
        self.setLayout(self.Vlayout)

        # style settings for QFrame
        self.box = QtWidgets.QFrame() 
        self.box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.font = QFont()
        self.font.setPointSize(15) 

        # add combobox to allow user to select between different models
        self.cb = QComboBox() 
        # adds option as text string followed by value connected to that button
        self.cb.addItem("Convolutional Neural Net", 1)
        self.cb.addItem("Feedforward Neural Network", 2)
        self.cb.currentIndexChanged.connect(self.select_model)
        self.Vlayout.addWidget(self.cb)   

        # Text box to communicate output
        self.box_text = QVBoxLayout()
        self.box_text.setAlignment(Qt.AlignTop)
        # text changes when a combobox selection is changed
        progress_label = QtWidgets.QLabel("Select a model above, and press TRAIN MODEL!")
        progress_label.setFont(self.font)
        self.box_text.addWidget(progress_label)
        self.box.setLayout(self.box_text)

        # progress bar and adds buttons to layout
        self.bar = QProgressBar()
        self.bar.setMaximum(100)
        self.b2 = QPushButton("Train Model")
        self.b3 = QPushButton("Test Model")
        self.bMnist = QPushButton("MNIST Examples")
        self.canvasButton = QPushButton("Canvas")
        self.canvasButton.clicked.connect(self.drawingButton)
        self.b4 = QPushButton("Exit")
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.b2)
        self.button_layout.addWidget(self.b3)
        self.button_layout.addWidget(self.canvasButton)
        self.button_layout.addWidget(self.bMnist)
        self.button_layout.addWidget(self.b4)
        # disables the following two buttons
        self.b3.setEnabled(False)
        self.canvasButton.setEnabled(False)
        self.Vlayout.addWidget(self.box)
        self.Vlayout.addWidget(self.bar)
        self.Vlayout.addLayout(self.button_layout)
    
        # display the team logo repeated
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

    # connected to the combobox which updates what model is being used
    def select_model(self,i):
        global model_choice
        model_choice = self.cb.itemData(i) # extracts value from combobox
        self.b2.setEnabled(True) # enables train button 
        self.b3.setEnabled(False) # disables test button
        self.canvasButton.setEnabled(False) # disables canvas button
        self.clear()
        # clears it a second time in case a QThread is running as it will output a new text in the text_box
        # this happens as clear() terminates a QThread if it's running, leading to the thread emitting a 
        # finished signal which is connected to a different function
        self.clear() 
        model_name = self.cb.itemText(i) # extracts text from combobox selection
        newText = "You have selected the " + model_name + \
        " model. Press the TRAIN MODEL button below to continue."
        newModel = QLabel(newText)
        newModel.setFont(self.font)
        newModel.setWordWrap(True) # wraps text to next line to fit window
        self.box_text.addWidget(newModel)
    
    # connected to the train model button, calls train() function from model.py
    def train_model(self):
        self.b2.setEnabled(False) # diables train button
        label_train = QtWidgets.QLabel("Training the Model... meanwhile, see examples of MNIST below.")
        label_train.setFont(self.font)
        self.box_text.addWidget(label_train)
        # when the user would change the model selected and completed training, the label above
        # would print multiple times
        # the loop below was used to fix that 
        for i in range(3,18):
            if self.box_text.itemAt(i) is not None:
                    self.box_text.itemAt(i).widget().deleteLater()
        self.train_thread.task_fin.connect(self.setValue)
        self.train_thread.start() # starts training QThread
        self.train_thread.finished.connect(self.updateTrain) # connects train QThread ending to function
        
    def setValue(self, value):
        # sets progress bar to signal value emitted by QThread
        self.bar.setValue(value)

    # once training is complete, this function performs updates to the GUI like disabling the TRAIN button
    def updateTrain(self):
        label_train_cmp = QtWidgets.QLabel("Model training complete! Press TEST MODEL to see results.")
        label_train_cmp.setFont(self.font)
        self.box_text.addWidget(label_train_cmp)
        self.b3.setEnabled(True) # enables test button
    
    # Connected to the test model button
    def test_model(self):
        label_test = QtWidgets.QLabel("Testing the Model...")
        label_test.setFont(self.font)
        self.box_text.addWidget(label_test)
        self.test_thread.task_fin.connect(self.setValue)
        self.test_thread.start() # starts testing QThread
        self.test_thread.finished.connect(self.updateTest) # connects test QThread ending to function

    # updates made to the GUI once testing is complete
    def updateTest(self):
        global Accuracy1
        # printing accuracy of model
        label_test_cmp = QLabel("Testing complete. Model Accuracy: %0.2f%% " % Accuracy1)
        label_test_cmp.setFont(self.font)
        self.box_text.addWidget(label_test_cmp)
        # the reason for this loop is the same as in updateTrain()
        for i in range(5,18):
            if self.box_text.itemAt(i) is not None:
                    self.box_text.itemAt(i).widget().deleteLater()
        # disables train and test button, enables canvas button
        self.b2.setEnabled(False)
        self.b3.setEnabled(False)
        self.canvasButton.setEnabled(True)

    # new button to link to the next window, the drawing canvas
    def drawingButton(self):
        self.newCanvas = CanvasWindow() # initiates drawing canvas window class instance
        self.newCanvas.initUI()  # displays new window with drawing canvas and prediction screen

    # connected to the view MNIST examples button
    def viewExample(self):
        self.exampleWin = MNISTImages()
        self.exampleWin.initExample() # opens new window with examples

    def clear(self):
        # clearing all text in the text_box layout
        for i in range(0,10):
            # makes sure widget is present before deleting
            if self.box_text.itemAt(i) is not None:
                self.box_text.itemAt(i).widget().deleteLater()
        # checks if threads are running and terminates if true
        if self.train_thread.isRunning():
            self.train_thread.terminate()
        if self.test_thread.isRunning():
            self.test_thread.terminate()
        self.bar.setValue(0)

    def clickExit(self):
        # checks if threads are running and terminates if true
        if self.train_thread.isRunning():
            self.train_thread.terminate()
        if self.test_thread.isRunning():
            self.test_thread.terminate()
        self.close() # closes UI

class TrainThread(QThread):
    # Qthread allows the function to run in the background while the user continues using the UI
    task_fin = pyqtSignal(int) # int signal emitted from QThread
    def run(self):
        global model_choice
        self.task_fin.emit(1) # emits 1% to progress bar to show to user that progress is happening
        for epoch in range(1,21):  # 20 epochs
            train(epoch = epoch, input = model_choice)
            self.task_fin.emit(epoch*5) # progress bar updates in steps of 5%

# global accuracy to be used between classes
Accuracy1 = 0     
class TestThread(QThread):
    # testing QThread for test button
    task_fin = pyqtSignal(int) # int signal emitted from QThread
    def run(self):
        # initialising global variables
        global model_choice
        global Accuracy1
        tmp_acc, _ = test(input = model_choice)
        Accuracy1 = tmp_acc*100 # gets accuracy of model in a % form
        self.task_fin.emit(100) # progress bar @100 when testing is complete

# global predicted number variable for use in other classes
predicted_num = 9999
class recogThread(QThread):
    task_fin = pyqtSignal(int) # int signal emitted from QThread
    def run(self):
        # initialising global variables
        global predicted_num
        global model_choice
        # recognises model using model selected from combobox
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