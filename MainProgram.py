#%% Import Libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QApplication,QMessageBox
import winsound
import numpy as np
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import os
import itertools


from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#%% Methods
# This method extracts features from the music file and returns the features
def construct_feature_list(fname,y,sr):
    features = [fname,np.average(librosa.feature.zero_crossing_rate(y)),np.median(librosa.feature.zero_crossing_rate(y)),np.var(librosa.feature.zero_crossing_rate(y)),
                    np.average(librosa.feature.spectral_bandwidth(y,sr)),np.median(librosa.feature.spectral_bandwidth(y,sr)),np.var(librosa.feature.spectral_bandwidth(y,sr)),
                        np.average(librosa.feature.rmse(y)),np.median(librosa.feature.rmse(y)),np.var(librosa.feature.rmse(y)),
                           np.average(librosa.feature.spectral_centroid(y,sr)),np.median(librosa.feature.spectral_centroid(y,sr)),np.var(librosa.feature.spectral_centroid(y,sr)),
                                np.average(librosa.feature.spectral_rolloff(y,sr)),np.median(librosa.feature.spectral_rolloff(y,sr)),np.var(librosa.feature.spectral_rolloff(y,sr)),
                                   np.average(librosa.feature.spectral_contrast(y,sr)),np.median(librosa.feature.spectral_contrast(y,sr)),np.var(librosa.feature.spectral_contrast(y,sr))]
    

    #calculate and add chroma_cqt
    tmp=librosa.feature.chroma_cqt(y,sr)
    tmpp=[None]*12
    for i in range(12):
        tmpp[i]=np.average(tmp[i])
    features=np.append(features,tmpp)

    #calculate and add mfcc
    tmp=librosa.feature.mfcc(y,sr)
    tmpp=[None]*20

    for i in range(20):
        tmpp[i]=np.average(tmp[i])
    features=np.append(features,tmpp)

    #calculate and add tonnetz
    tmp=librosa.feature.tonnetz(y,sr)
    tmpp=[None]*18
    i=0
    j=0
    while i<16:
        tmpp[i]=np.average(tmp[j])
        tmpp[i+1]=np.median(tmp[j])
        tmpp[i+2]=np.var(tmp[j])
        i=i+3
        j=j+1
    features=np.append(features,tmpp)
    #calculate and add spectral flatness
    tmpp=[np.average(librosa.feature.spectral_flatness(y)),np.median(librosa.feature.spectral_flatness(y)),
                  np.var(librosa.feature.spectral_flatness(y))
                  ]
    features=np.append(features,tmpp)
    #calculate and add poly_features
    coeff=librosa.feature.poly_features(y,sr)
    tmpp=[np.average(coeff[0]),np.median(coeff[0]),np.var(coeff[0])]
    features=np.append(features,tmpp)

    #new features
    #calculate and add  varmfcc
    tmp=librosa.feature.mfcc(y,sr)
    tmpp=[None]*20
    n_mfcc=20
    mfcc_col_size = len(tmp[0])
    for i in range(20):
        tmpp[i]=np.var(tmp[i])
    features=np.append(features,tmpp)
    
    #MFCC derivative feature extraction
    mfcc_derivative = np.empty([n_mfcc, mfcc_col_size], dtype=float)
    
    for i in range(n_mfcc):
        mfcc_derivative[i][0] = 0
        
    for i in range(n_mfcc):
        for j in range(1, mfcc_col_size):  
            mfcc_derivative[i][j] = tmp[i][j] - tmp[i][j-1]
            
    # We append mean value of the mfcc derivatives
    mfcc_derivative_avgs_stds = np.empty(2*n_mfcc, dtype=float)
    j=0  
    for i in range(n_mfcc):
        mfcc_derivative_avgs_stds[j] = np.average(mfcc_derivative[i])  
        mfcc_derivative_avgs_stds[j+1] = np.var(mfcc_derivative[i])
        j = j+2
    features = np.append(features, mfcc_derivative_avgs_stds)

    #calculate and add var chroma_cqt
    tmp=librosa.feature.chroma_cqt(y,sr)
    tmpp=[None]*12
    for i in range(12):
        tmpp[i]=np.var(tmp[i])
    features=np.append(features,tmpp)

    return features


class Ui_MainWindow(object):
    
    Genres={"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4
        ,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9
        }
    InverseGenres={0:"blues",1:"classical",2:"country",3:"disco",4:"hiphop"
               ,5:"jazz",6:"metal",7:"pop",8:"reggae",9:"rock"
               }
    data=data = [random.random() for i in range(25)]
    data = np.zeros(1)
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(460, 734)
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(460, 734)
        MainWindow.move(400,0)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(460, 734))
        MainWindow.setMaximumSize(QtCore.QSize(460, 734))    
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(139, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(75, 153, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(7, 78, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(133, 186, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(139, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(75, 153, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(7, 78, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(133, 186, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(139, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(75, 153, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(7, 78, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(5, 58, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(11, 117, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        MainWindow.setPalette(palette)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 441, 711))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        self.tabWidget.setPalette(palette)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setStyleSheet("background-color: rgb(11, 117, 255);")
        self.tab.setObjectName("tab")
        self.l_confusion_matrix = QtWidgets.QLabel(self.tab)
        self.l_confusion_matrix.setGeometry(QtCore.QRect(10, 316, 411, 361))
        self.l_confusion_matrix.setMinimumSize(QtCore.QSize(347, 297))
        self.l_confusion_matrix.setMaximumSize(QtCore.QSize(444, 397))
        self.l_confusion_matrix.setBaseSize(QtCore.QSize(344, 294))
        self.l_confusion_matrix.setStyleSheet("background-color: rgb(255, 255, 255);border: 1px solid;")
        self.l_confusion_matrix.setText("")
        self.l_confusion_matrix.setAlignment(QtCore.Qt.AlignCenter)
        self.l_confusion_matrix.setObjectName("l_confusion_matrix")
        self.ZeroCrossBox = QtWidgets.QCheckBox(self.tab)
        self.ZeroCrossBox.setGeometry(QtCore.QRect(20, 20, 121, 17))
        self.ZeroCrossBox.setObjectName("ZeroCrossBox")
        self.RMSEBox = QtWidgets.QCheckBox(self.tab)
        self.RMSEBox.setGeometry(QtCore.QRect(20, 80, 70, 17))
        self.RMSEBox.setObjectName("RMSEBox")
        self.TonnetzBox = QtWidgets.QCheckBox(self.tab)
        self.TonnetzBox.setGeometry(QtCore.QRect(20, 110, 70, 17))
        self.TonnetzBox.setObjectName("TonnetzBox")
        self.MFCCBox = QtWidgets.QCheckBox(self.tab)
        self.MFCCBox.setGeometry(QtCore.QRect(150, 80, 70, 17))
        self.MFCCBox.setObjectName("MFCCBox")
        self.SpecBadwithBox = QtWidgets.QCheckBox(self.tab)
        self.SpecBadwithBox.setGeometry(QtCore.QRect(20, 50, 111, 17))
        self.SpecBadwithBox.setObjectName("SpecBadwithBox")
        self.SpecRollOfBox = QtWidgets.QCheckBox(self.tab)
        self.SpecRollOfBox.setGeometry(QtCore.QRect(150, 50, 101, 17))
        self.SpecRollOfBox.setObjectName("SpecRollOfBox")
        self.SpecFlatBox = QtWidgets.QCheckBox(self.tab)
        self.SpecFlatBox.setGeometry(QtCore.QRect(270, 50, 111, 17))
        self.SpecFlatBox.setObjectName("SpecFlatBox")
        self.DMFCCBox = QtWidgets.QCheckBox(self.tab)
        self.DMFCCBox.setGeometry(QtCore.QRect(270, 80, 101, 17))
        self.DMFCCBox.setObjectName("DMFCCBox")
        self.CQTBox = QtWidgets.QCheckBox(self.tab)
        self.CQTBox.setGeometry(QtCore.QRect(150, 110, 91, 17))
        self.CQTBox.setObjectName("CQTBox")
        self.PolyBox = QtWidgets.QCheckBox(self.tab)
        self.PolyBox.setGeometry(QtCore.QRect(270, 110, 131, 17))
        self.PolyBox.setObjectName("PolyBox")
        self.SpecontrastBox = QtWidgets.QCheckBox(self.tab)
        self.SpecontrastBox.setGeometry(QtCore.QRect(270, 20, 111, 17))
        self.SpecontrastBox.setObjectName("SpecontrastBox")
        self.SpecCentroidBox = QtWidgets.QCheckBox(self.tab)
        self.SpecCentroidBox.setGeometry(QtCore.QRect(150, 20, 111, 17))
        self.SpecCentroidBox.setObjectName("SpecCentroidBox")
        self.comboBox = QtWidgets.QComboBox(self.tab)
        self.comboBox.setGeometry(QtCore.QRect(150, 160, 151, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"selection-color: rgb(0, 0, 0);\n"
"selection-background-color: rgb(255, 255, 255);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.Kspin = QtWidgets.QSpinBox(self.tab)
        self.Kspin.setGeometry(QtCore.QRect(20, 160, 42, 22))
        self.Kspin.setMinimum(1)
        self.Kspin.setSingleStep(2)        
        self.Kspin.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Kspin.setObjectName("KSpin")
        self.KFoldSpin = QtWidgets.QSpinBox(self.tab)
        self.KFoldSpin.setGeometry(QtCore.QRect(80, 160, 42, 22))
        self.KFoldSpin.setMinimum(2)        
        self.KFoldSpin.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.KFoldSpin.setObjectName("KFoldSpin")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(30, 140, 21, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(90, 140, 31, 20))
        self.label_5.setObjectName("label_5")
        self.classificationButton = QtWidgets.QPushButton(self.tab)
        self.classificationButton.setGeometry(QtCore.QRect(20, 200, 381, 23))
        self.classificationButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.classificationButton.setObjectName("classificationButton")
        self.PrecisionRateLine = QtWidgets.QLineEdit(self.tab)
        self.PrecisionRateLine.setEnabled(False)
        self.PrecisionRateLine.setGeometry(QtCore.QRect(10, 290, 61, 20))
        self.PrecisionRateLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.PrecisionRateLine.setDragEnabled(False)
        self.PrecisionRateLine.setObjectName("PrecisionRateLine")
        self.accuracyRateLine = QtWidgets.QLineEdit(self.tab)
        self.accuracyRateLine.setEnabled(False)
        self.accuracyRateLine.setGeometry(QtCore.QRect(240, 290, 91, 20))
        self.accuracyRateLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.accuracyRateLine.setDragEnabled(False)
        self.accuracyRateLine.setObjectName("accuracyRateLine")
        self.maxRateLine = QtWidgets.QLineEdit(self.tab)
        self.maxRateLine.setEnabled(False)
        self.maxRateLine.setGeometry(QtCore.QRect(340, 290, 91, 20))
        self.maxRateLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.maxRateLine.setDragEnabled(False)
        self.maxRateLine.setObjectName("maxRateLine")
        self.RecallRateLine = QtWidgets.QLineEdit(self.tab)
        self.RecallRateLine.setEnabled(False)
        self.RecallRateLine.setGeometry(QtCore.QRect(80, 290, 51, 20))
        self.RecallRateLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.RecallRateLine.setDragEnabled(False)
        self.RecallRateLine.setObjectName("RecallRateLine")
        self.minRateLine = QtWidgets.QLineEdit(self.tab)
        self.minRateLine.setEnabled(False)
        self.minRateLine.setGeometry(QtCore.QRect(140, 290, 91, 20))
        self.minRateLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.minRateLine.setDragEnabled(False)
        self.minRateLine.setObjectName("minRateLine")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(10, 270, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(80, 270, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(140, 270, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self.tab)
        self.label_6.setGeometry(QtCore.QRect(240, 270, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab)
        self.label_7.setGeometry(QtCore.QRect(340, 270, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab)
        self.label_8.setGeometry(QtCore.QRect(30, 230, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setStyleSheet("background-color: rgb(11, 117, 255);")
        self.tab_2.setObjectName("tab_2")
        self.musicFileName = QtWidgets.QLineEdit(self.tab_2)
        self.musicFileName.setEnabled(False)
        self.musicFileName.setGeometry(QtCore.QRect(10, 20, 321, 20))
        self.musicFileName.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.musicFileName.setDragEnabled(False)
        self.musicFileName.setReadOnly(False)
        self.musicFileName.setObjectName("musicFileName")
        self.openButton = QtWidgets.QPushButton(self.tab_2)
        self.openButton.setGeometry(QtCore.QRect(340, 20, 91, 21))
        self.openButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.openButton.setObjectName("openButton")
        self.playButton = QtWidgets.QPushButton(self.tab_2)
        self.playButton.setGeometry(QtCore.QRect(10, 60, 151, 23))
        self.playButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.playButton.setObjectName("playButton")
        self.stopButton = QtWidgets.QPushButton(self.tab_2)
        self.stopButton.setGeometry(QtCore.QRect(180, 60, 151, 23))
        self.stopButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.stopButton.setObjectName("stopButton")
        self.Kspin2 = QtWidgets.QSpinBox(self.tab_2)
        self.Kspin2.setGeometry(QtCore.QRect(340, 60, 42, 22))
        self.Kspin2.setMinimum(1)
        self.Kspin2.setSingleStep(2)
        self.Kspin2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Kspin2.setObjectName("KSpin2")
        self.KFoldSpin2 = QtWidgets.QSpinBox(self.tab_2)
        self.KFoldSpin2.setGeometry(QtCore.QRect(390, 60, 42, 22))
        self.KFoldSpin2.setMinimum(2)        
        self.KFoldSpin2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.KFoldSpin2.setObjectName("KFoldSpin2")
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setGeometry(QtCore.QRect(350, 40, 21, 20))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_2)
        self.label_10.setGeometry(QtCore.QRect(390, 40, 41, 20))
        self.label_10.setObjectName("label_10")
        self.ZeroCrossBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.ZeroCrossBox2.setGeometry(QtCore.QRect(10, 100, 121, 17))
        self.ZeroCrossBox2.setObjectName("ZeroCrossBox2")
        self.SpecCentroidBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.SpecCentroidBox2.setGeometry(QtCore.QRect(150, 100, 111, 17))
        self.SpecCentroidBox2.setObjectName("SpecCentroidBox2")
        self.SpecontrastBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.SpecontrastBox2.setGeometry(QtCore.QRect(280, 100, 111, 17))
        self.SpecontrastBox2.setObjectName("SpecontrastBox2")
        self.SpecBadwithBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.SpecBadwithBox2.setGeometry(QtCore.QRect(10, 130, 111, 17))
        self.SpecBadwithBox2.setObjectName("SpecBadwithBox2")
        self.SpecRollOfBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.SpecRollOfBox2.setGeometry(QtCore.QRect(150, 130, 101, 17))
        self.SpecRollOfBox2.setObjectName("SpecRollOfBox2")
        self.SpecFlatBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.SpecFlatBox2.setGeometry(QtCore.QRect(280, 130, 111, 17))
        self.SpecFlatBox2.setObjectName("SpecFlatBox2")
        self.RMSEBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.RMSEBox2.setGeometry(QtCore.QRect(10, 160, 70, 17))
        self.RMSEBox2.setObjectName("RMSEBox2")
        self.MFCCBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.MFCCBox2.setGeometry(QtCore.QRect(150, 160, 70, 17))
        self.MFCCBox2.setObjectName("MFCCBox2")
        self.DMFCCBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.DMFCCBox2.setGeometry(QtCore.QRect(280, 160, 101, 17))
        self.DMFCCBox2.setObjectName("DMFCCBox2")
        self.TonnetzBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.TonnetzBox2.setGeometry(QtCore.QRect(10, 190, 70, 17))
        self.TonnetzBox2.setObjectName("TonnetzBox2")
        self.CQTBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.CQTBox2.setGeometry(QtCore.QRect(150, 190, 91, 17))
        self.CQTBox2.setObjectName("CQTBox2")
        self.PolyBox2 = QtWidgets.QCheckBox(self.tab_2)
        self.PolyBox2.setGeometry(QtCore.QRect(280, 190, 131, 17))
        self.PolyBox2.setObjectName("PolyBox2")
        self.comboBox2 = QtWidgets.QComboBox(self.tab_2)
        self.comboBox2.setGeometry(QtCore.QRect(260, 230, 161, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox2.setFont(font)
        self.comboBox2.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"selection-color: rgb(0, 0, 0);\n"
"selection-background-color: rgb(255, 255, 255);")
        self.comboBox2.setObjectName("comboBox2")
        self.comboBox2.addItem("")
        self.comboBox2.addItem("")
        self.comboBox2.addItem("")
        self.comboBox2.addItem("")
        self.comboBox2.addItem("")
        self.FindGenreButton = QtWidgets.QPushButton(self.tab_2)
        self.FindGenreButton.setGeometry(QtCore.QRect(10, 230, 241, 23))
        self.FindGenreButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.FindGenreButton.setObjectName("FindGenreButton")
        self.genreLabel = QtWidgets.QLabel(self.tab_2)
        self.genreLabel.setGeometry(QtCore.QRect(10, 270, 341, 31))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.genreLabel.setFont(font)
        self.genreLabel.setObjectName("genreLabel")
        self.genreLine = QtWidgets.QLineEdit(self.tab_2)
        self.genreLine.setEnabled(False)
        self.genreLine.setGeometry(QtCore.QRect(20, 310, 391, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.genreLine.setFont(font)
        self.genreLine.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.genreLine.setDragEnabled(False)
        self.genreLine.setObjectName("genreLine")
        self.l_song_figure = QtWidgets.QLabel(self.tab_2)
        self.l_song_figure.setGeometry(QtCore.QRect(20, 370, 386, 278))
        self.l_song_figure.setMinimumSize(QtCore.QSize(386, 278))
        self.l_song_figure.setMaximumSize(QtCore.QSize(386, 278))
        self.l_song_figure.setBaseSize(QtCore.QSize(386, 278))
        self.l_song_figure.setStyleSheet("border: 1px solid;\n"
"background-color: rgb(255, 255, 255);")
        self.l_song_figure.setText("")
        self.l_song_figure.setObjectName("l_song_figure")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.openButton.clicked.connect(self.openFile)
        self.playButton.clicked.connect(self.playMusic)
        self.stopButton.clicked.connect(self.stopMusic)
        self.FindGenreButton.clicked.connect(self.FindGenre)
        self.classificationButton.clicked.connect(self.Classify)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Finding Genre of Music Program"))
        self.ZeroCrossBox.setText(_translate("MainWindow", "Zero Crossing Rate"))
        self.RMSEBox.setText(_translate("MainWindow", "RMSE"))
        self.TonnetzBox.setText(_translate("MainWindow", "Tonnetz"))
        self.MFCCBox.setText(_translate("MainWindow", "MFCC"))
        self.SpecBadwithBox.setText(_translate("MainWindow", "Spectral Bandwith"))
        self.SpecRollOfBox.setText(_translate("MainWindow", "Spectral Rollof"))
        self.SpecFlatBox.setText(_translate("MainWindow", "Spectral Flatness"))
        self.DMFCCBox.setText(_translate("MainWindow", "MFCC Derivative"))
        self.CQTBox.setText(_translate("MainWindow", "Chroma CQT"))
        self.PolyBox.setText(_translate("MainWindow", "Poly_Features"))
        self.SpecontrastBox.setText(_translate("MainWindow", "Spectral Contrast"))
        self.SpecCentroidBox.setText(_translate("MainWindow", "Spectral Centroid"))
        self.comboBox.setItemText(0, _translate("MainWindow", "KNN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Decision Tree"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Naive Bayes"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Random Forest"))
        self.comboBox.setItemText(4, _translate("MainWindow", "SVM"))
        self.label_4.setText(_translate("MainWindow", "   K"))
        self.label_5.setText(_translate("MainWindow", "KFold"))
        self.classificationButton.setText(_translate("MainWindow", "START CLASSIFICATION"))
        self.label.setText(_translate("MainWindow", " Precision"))
        self.label_2.setText(_translate("MainWindow", " Recall"))
        self.label_3.setText(_translate("MainWindow", "Min Accuracy"))
        self.label_6.setText(_translate("MainWindow", "Avg Accuracy"))
        self.label_7.setText(_translate("MainWindow", "Max Accuracy"))
        self.label_8.setText(_translate("MainWindow", "Accuracy of Selected Algorithm:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Classification"))
        self.openButton.setText(_translate("MainWindow", "OPEN"))
        self.playButton.setText(_translate("MainWindow", "PLAY"))
        self.stopButton.setText(_translate("MainWindow", "STOP"))
        self.label_9.setText(_translate("MainWindow", "   K"))
        self.label_10.setText(_translate("MainWindow", "   KFold"))
        self.ZeroCrossBox2.setText(_translate("MainWindow", "Zero Crossing Rate"))
        self.SpecCentroidBox2.setText(_translate("MainWindow", "Spectral Centroid"))
        self.SpecontrastBox2.setText(_translate("MainWindow", "Spectral Contrast"))
        self.SpecBadwithBox2.setText(_translate("MainWindow", "Spectral Bandwith"))
        self.SpecRollOfBox2.setText(_translate("MainWindow", "Spectral Rollof"))
        self.SpecFlatBox2.setText(_translate("MainWindow", "Spectral Flatness"))
        self.RMSEBox2.setText(_translate("MainWindow", "RMSE"))
        self.MFCCBox2.setText(_translate("MainWindow", "MFCC"))
        self.DMFCCBox2.setText(_translate("MainWindow", "MFCC Derivative"))
        self.TonnetzBox2.setText(_translate("MainWindow", "Tonnetz"))
        self.CQTBox2.setText(_translate("MainWindow", "Chroma CQT"))
        self.PolyBox2.setText(_translate("MainWindow", "Poly_Features"))
        self.comboBox2.setItemText(0, _translate("MainWindow", "KNN"))
        self.comboBox2.setItemText(1, _translate("MainWindow", "Decision Tree"))
        self.comboBox2.setItemText(2, _translate("MainWindow", "Naive Bayes"))
        self.comboBox2.setItemText(3, _translate("MainWindow", "Random Forest"))
        self.comboBox2.setItemText(4, _translate("MainWindow", "SVM"))
        self.FindGenreButton.setText(_translate("MainWindow", "Find Genre of Selected Music"))
        self.genreLabel.setText(_translate("MainWindow", "Genre of Selected Music :"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Genre Estimation"))

    # This method opens the music file
    def openFile(self):
        fileName=QFileDialog.getOpenFileName(caption='Open file',filter="Music files (*.mp3 *.wav *.au)")
        self.musicFileName.setText(fileName[0])

    # User can play the music with this method
    def playMusic(self):
        winsound.PlaySound(self.musicFileName.text(),winsound.SND_ASYNC)

    # User can stop the music with this method
    def stopMusic(self):
        winsound.PlaySound(None,winsound.SND_PURGE)

    # The most important method of the program
    # It predicts the genre of the selected music by using selected Machine Learning Algorithm
    def FindGenre(self):
        algorithmName = self.comboBox2.currentText()
        # We have to know the algorithm name which selected by the user
        # There are five algorithms in the system
        # They are ;
        # -Decision Tree
        # - KNN
        # - SVM (Support Vector Machine)
        # - Random Forest
        # -Naive Bayes
        if(algorithmName=="Decision Tree"):
            model = DecisionTreeClassifier()
        if(algorithmName=="KNN"):
            k = (self.Kspin2.value())
            if(k % 2 == 0 ):
                k = k + 1
                #if k value is even,we should convert it to odd number
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        if(algorithmName=="SVM"):
            model =  SVC(kernel='linear',random_state=42)
        if(algorithmName=="Random Forest"):
            model = RandomForestClassifier(criterion='entropy', random_state=42)
        if(algorithmName=="Naive Bayes"):
             model = GaussianNB()
             
        #we found the selected algorithm
        #let's read the music data
        # If the user haven't selected a music yet, we show an warning message to the user
        if(self.musicFileName.text()!="" and self.musicFileName.text()!="Please Choose a Music File"):
            if(self.ZeroCrossBox2.checkState()==0 and self.SpecBadwithBox2.checkState()==0 and self.RMSEBox2.checkState()==0 and self.SpecCentroidBox2.checkState()==0 and self.SpecRollOfBox2.checkState()==0 and self.SpecontrastBox2.checkState()==0 and self.CQTBox2.checkState()==0 and self.MFCCBox2.checkState()==0 and self.DMFCCBox2.checkState()==0 and self.TonnetzBox2.checkState()==0 and self.SpecFlatBox2.checkState()==0 and self.PolyBox2.checkState()==0):
                print("mesaj box")
                msgBox=QMessageBox ()
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("Please Choose The Features          ")
                msgBox.setWindowTitle("Warning!!!")
                ret = msgBox.exec_()
                return

            # We use the librosa library to read the music file
            musicData,sr=librosa.load(self.musicFileName.text())
            
            # After the reading music file, We can extract features from the music
            features = list(construct_feature_list(self.musicFileName.text(),musicData,sr))
            
            remain_features=[]

            # User can select features from gui, which will be used in algorithms
            # We should know selected features to use them, the below side is related it
            if(self.ZeroCrossBox2.checkState()==0 or self.SpecBadwithBox2.checkState()==0 or self.RMSEBox2.checkState()==0 or self.SpecCentroidBox2.checkState()==0 or self.SpecRollOfBox2.checkState()==0 or self.SpecontrastBox2.checkState()==0 or self.CQTBox2.checkState()==0 or self.MFCCBox2.checkState()==0 or self.DMFCCBox2.checkState()==0 or self.TonnetzBox2.checkState()==0 or self.SpecFlatBox2.checkState()==0 or self.PolyBox2.checkState()==0):
                if(self.ZeroCrossBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[1:4])
                if(self.SpecBadwithBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[4:7])
                if(self.RMSEBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[7:10])
                if(self.SpecCentroidBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[10:13])
                if(self.SpecRollOfBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[13:16])
                if(self.SpecontrastBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[16:19])
                if(self.CQTBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[19:31])
                if(self.MFCCBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[31:51])
                if(self.TonnetzBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[51:69])
                if(self.SpecFlatBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[69:72])
                if(self.PolyBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[72:75])
                if(self.MFCCBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[75:95])
                if(self.DMFCCBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[95:135])
                if(self.CQTBox2.checkState()!=0):
                    remain_features = np.append(remain_features, features[135:])
            else:
                # First element of features is the name,we don't need the name machine learning
                remain_features = features[1:]

            #Let's guess the genre of selected music
            valuesFromDb = self.takeFeaturesForGenreEstimation()
            query = "select "+valuesFromDb+" from Features"
            data = pd.read_sql_query(query,con)
            data.drop(["Name"],axis=1,inplace=True)
            model1,scaler,acc,prec,recall,max_acc,min_acc = self.Find_Accuracy(model,1)
            # We created the model to predict the genre of the music
            
            # Draw the figure of music
            m=len(remain_features)
            data.drop(["Genre"],axis=1,inplace=True)
            df = pd.DataFrame(np.array(remain_features).reshape(1,m), columns = list(data.columns.values))
            
            # we have to convert numpy array to dataframe to use in machine learning algorithm
            fig1=plt.figure()
            librosa.display.waveplot(musicData, sr=sr)
            plt.title('Waveplot of the Song')
            plt.savefig('fig2.png', bbox_inches="tight", pad_inches=0.3)
            self.l_song_figure.setPixmap(QtGui.QPixmap('fig2.png').scaled(386, 278, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation))
            plt.clf()

            # Normalization is necessary to improve algorithms' performance, specially in KNN
            X_test  = scaler.transform(df)            
            Y_pred = model1.predict(X_test)
            
            #we found the genre of selected music
            predict_genre = self.InverseGenres[int(Y_pred)]
            
            #let's print the result on screen
            self.genreLine.setText(str(predict_genre))
            msgBox1=QMessageBox ()
            msgBox1.setIcon(QMessageBox.Information)
            msgBox1.setText("The Results are Ready     ")
            msgBox1.setWindowTitle("Information   ")
            ret = msgBox1.exec_()                
            return
        else:
            self.musicFileName.setText("Please Choose a Music File")
            return

    # This method creates a model by using selected algorithm
    # And return this model and performance evaluation 
    def Classify(self):
        algorithmName = self.comboBox.currentText()
        
        # Which algorithm is selected, we should check it
        if(algorithmName=="Decision Tree"):
            model = DecisionTreeClassifier()
        if(algorithmName=="KNN"):
            k = (self.Kspin.value())
            if(k % 2 == 0 ):
                k = k + 1
                #if k value is even,we should convert it to odd number
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        if(algorithmName=="SVM"):
            model =  SVC(kernel='linear',random_state=42)
        if(algorithmName=="Random Forest"):
            model = RandomForestClassifier(criterion='entropy', random_state=42)
        if(algorithmName=="Naive Bayes"):
             model = GaussianNB()

        # We found the selected algorithm
        # User should select at least one feature, if user does not select any feature we can't predict the genre of the music
        if(self.ZeroCrossBox.checkState()==0 and self.SpecBadwithBox.checkState()==0 and self.RMSEBox.checkState()==0 and self.SpecCentroidBox.checkState()==0 and self.SpecRollOfBox.checkState()==0 and self.SpecontrastBox.checkState()==0 and self.CQTBox.checkState()==0 and self.MFCCBox.checkState()==0 and self.DMFCCBox.checkState()==0 and self.TonnetzBox.checkState()==0 and self.SpecFlatBox.checkState()==0 and self.PolyBox.checkState()==0):
            msgBox=QMessageBox ()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Please Choose The Features          ")
            msgBox.setWindowTitle("Warning!!!")
            ret = msgBox.exec_()
            
            return
        
        # Find accuracy rates for our model
        model1,scaler,acc,prec,recall,max_acc,min_acc = self.Find_Accuracy(model,0)

        # We created the model and found the performance results
        # Let's show them on the screen
        self.accuracyRateLine.setText(str(round(acc,2)))
        self.PrecisionRateLine.setText(str(round(prec,2)))
        self.RecallRateLine.setText(str(round(recall,2)))
        self.maxRateLine.setText(str(round(max_acc,2)))
        self.minRateLine.setText(str(round(min_acc,2)))
        msgBox1=QMessageBox ()
        msgBox1.setIcon(QMessageBox.Information)
        msgBox1.setText("The Results are Ready     ")
        msgBox1.setWindowTitle("Information   ")
        ret = msgBox1.exec_()                
        return

    # This method was written to find selected features and returns their names
    def takeFeaturesForGenreEstimation(self):
        columns = "Name,"
        MFCCSelect=0
        CQTSelect=0
        if(self.ZeroCrossBox2.checkState()!=0):
            columns = columns+"MeanZeroCross,MedianZeroCross,VarZeroCross,"
        if(self.SpecBadwithBox2.checkState()!=0):
            columns = columns +"MeanSpecBand,MedianSpecBand,VarSpecBand,"
        if(self.RMSEBox2.checkState()!=0):
            columns = columns +"MeanRmse,MedianRmse,VarRmse,"
        if(self.SpecCentroidBox2.checkState()!=0):
            columns = columns +"MeanSpecCentroid,MedianSpecCentroid,VarSpecCentroid,"
        if(self.SpecRollOfBox2.checkState()!=0):
            columns = columns + "MeanSpecRoll,MedianSpecRoll,VarSpecRoll,"
        if(self.SpecontrastBox2.checkState()!=0):
            columns = columns + "MeanSpecContrast,MedianSpecContrast,VarSpecContrast,"
        if(self.CQTBox2.checkState()!=0):
            CQTSelect=1
            for i in range(12):
                j=i+1
                strCqt="CQT"+str(j)
                columns = columns+strCqt+","
        if(self.MFCCBox2.checkState()!=0):
            for i in range(20):
                j=i+1
                strMFCC="MFCC"+str(j)
                columns = columns+strMFCC+","
            MFCCSelect=1
        if(self.TonnetzBox2.checkState()!=0):
            for i in range(6):
                j=i+1
                str1="MeanTonnetz"+str(j)
                str2="MedianTonnetz"+str(j)
                str3="VarTonnetz"+str(j)
                columns=columns+str1+","+str2+","+str3+","
        if(self.SpecFlatBox2.checkState()!=0):
            columns = columns + "MeanSpecFlatness,MedianSpecFlatness,VarSpecFlatness,"
        if(self.PolyBox2.checkState()!=0):
            columns = columns + "MeanPoly,MedianPoly,VarPoly,"
        if(MFCCSelect!=0):
            for i in range(20):
                j=i+1
                strMFCC="VarMFCC"+str(j)
                columns = columns+strMFCC+","
        if(self.DMFCCBox2.checkState()!=0):
            for i in range(20):
                j=i+1
                strMFCC="DMFCC"+str(j)
                strVarMFCC="VarDMFCC"+str(j)
                columns = columns+strMFCC+","+strVarMFCC+","
        if(CQTSelect!=0):
            for i in range(12):
                j=i+1
                strCqt="VarCQT"+str(j)
                columns = columns+strCqt+","

        columns = columns +"Genre"
        return columns

    # This method finds the success of the models
    def Find_Accuracy(self,model,forGenre):
        # for prediction of genre
        if(forGenre==1):
            valuesFromDb = self.takeFeaturesForGenreEstimation()
            kFold = self.KFoldSpin2.value()      
        else:
            # for genre classification
            valuesFromDb = self.takeFeatures()
            kFold = self.KFoldSpin.value()
            
        query = "select "+valuesFromDb+" from Features"
        data = pd.read_sql_query(query,con)
        data.drop(["Name"],axis=1,inplace=True)            
        newList=[]
        global cnf_matrix_total
        for i in data.Genre:
            newList.append(self.Genres[i])

        data.Genre =newList
        y = data.Genre.values
        x_data = data.drop(["Genre"],axis=1)

        # We applied Min Max Normalization to our data set
        scaler = preprocessing.MinMaxScaler((-1,1)) 
        random_state = 42

        # We use the K Fold Cross Validation
        # We read the K Fold number from the user
        cv = model_selection.KFold(n_splits=kFold, shuffle=True, random_state=random_state)
        max_acc = 0
        min_acc = 100
        total_acc = 0
        total_precision=0
        total_recall=0
        
        X=x_data
        scaler.fit(X)
        first = 1
        i = 0

        for train_index, test_index in cv.split(X.values):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            x_train = scaler.transform(x_train.values)
            x_test  = scaler.transform(x_test.values)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred)*100
            total_acc += acc
            prec = metrics.precision_score(y_test,y_pred,average='macro')*100
            total_precision +=prec
            recall = metrics.recall_score(y_test,y_pred,average='macro')*100
            total_recall +=recall
            cnf_matrix = confusion_matrix(y_test, y_pred)

            if acc > max_acc:
                max_pred = y_pred
                max_acc = acc
                max_cnf_matrix = cnf_matrix
            if min_acc > acc:
                min_acc = acc
            if first == 0:
                cnf_matrix_total = cnf_matrix_total + cnf_matrix
            else:
                cnf_matrix_total = cnf_matrix
                first = 0
            i += 1
        avg_acc = total_acc / kFold
        avg_prec = total_precision / kFold
        avg_recall = total_recall / kFold
        if(forGenre==0):
            self.plot_confusion_matrix(cnf_matrix_total, classes=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
            self.l_confusion_matrix.setPixmap(QtGui.QPixmap('fig.png'))
            
        return model,scaler,avg_acc,avg_prec,avg_recall,max_acc,min_acc
    
    def takeFeatures(self):
        columns = "Name,"
        MFCCSelect=0
        CQTSelect=0
        if(self.ZeroCrossBox.checkState()!=0):
            columns = columns+"MeanZeroCross,MedianZeroCross,VarZeroCross,"
        if(self.SpecBadwithBox.checkState()!=0):
            columns = columns +"MeanSpecBand,MedianSpecBand,VarSpecBand,"
        if(self.RMSEBox.checkState()!=0):
            columns = columns +"MeanRmse,MedianRmse,VarRmse,"
        if(self.SpecCentroidBox.checkState()!=0):
            columns = columns +"MeanSpecCentroid,MedianSpecCentroid,VarSpecCentroid,"
        if(self.SpecRollOfBox.checkState()!=0):
            columns = columns + "MeanSpecRoll,MedianSpecRoll,VarSpecRoll,"
        if(self.SpecontrastBox.checkState()!=0):
            columns = columns + "MeanSpecContrast,MedianSpecContrast,VarSpecContrast,"
        if(self.CQTBox.checkState()!=0):
            CQTSelect=1
            for i in range(12):
                j=i+1
                strCqt="CQT"+str(j)
                columns = columns+strCqt+","
        if(self.MFCCBox.checkState()!=0):
            for i in range(20):
                j=i+1
                strMFCC="MFCC"+str(j)
                columns = columns+strMFCC+","
            MFCCSelect=1
        if(self.TonnetzBox.checkState()!=0):
            for i in range(6):
                j=i+1
                str1="MeanTonnetz"+str(j)
                str2="MedianTonnetz"+str(j)
                str3="VarTonnetz"+str(j)
                columns=columns+str1+","+str2+","+str3+","
        if(self.SpecFlatBox.checkState()!=0):
            columns = columns + "MeanSpecFlatness,MedianSpecFlatness,VarSpecFlatness,"
        if(self.PolyBox.checkState()!=0):
            columns = columns + "MeanPoly,MedianPoly,VarPoly,"
        if(MFCCSelect!=0):
            for i in range(20):
                j=i+1
                strMFCC="VarMFCC"+str(j)
                columns = columns+strMFCC+","
        if(self.DMFCCBox.checkState()!=0):
            for i in range(20):
                j=i+1
                strMFCC="DMFCC"+str(j)
                strVarMFCC="VarDMFCC"+str(j)
                columns = columns+strMFCC+","+strVarMFCC+","
        if(CQTSelect!=0):
            for i in range(12):
                j=i+1
                strCqt="VarCQT"+str(j)
                columns = columns+strCqt+","

        columns = columns +"Genre"
        return columns

    # This method plots the confussiong matrix on the gui
    def plot_confusion_matrix(self, cm, classes):
        """ This function prints and plots the confusion matrix. """
        fig = plt.figure(figsize=(4, 4), dpi=100)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #plt.tight_layout()
        plt.savefig('fig.png', bbox_inches="tight", pad_inches=0.5)
        plt.clf()
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    con = sqlite3.connect("Project.db")
    cursor = con.cursor()    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

