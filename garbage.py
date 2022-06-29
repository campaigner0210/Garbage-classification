# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GarbageSorting.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GarbageSortingUI(object):
    def setupUi(self, GarbageSortingUI):
        GarbageSortingUI.setObjectName("GarbageSortingUI")
        GarbageSortingUI.resize(917, 671)
        GarbageSortingUI.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(GarbageSortingUI)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Input = QtWidgets.QPushButton(self.centralwidget)
        self.Input.setObjectName("Input")
        self.gridLayout.addWidget(self.Input, 2, 1, 1, 1)
        self.Output = QtWidgets.QPushButton(self.centralwidget)
        self.Output.setObjectName("Output")
        self.gridLayout.addWidget(self.Output, 3, 1, 1, 1)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 442, 552))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ShiBieJieGuo = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.ShiBieJieGuo.setGeometry(QtCore.QRect(10, 10, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.ShiBieJieGuo.setFont(font)
        self.ShiBieJieGuo.setObjectName("ShiBieJieGuo")
        self.TestResult = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents)
        self.TestResult.setGeometry(QtCore.QRect(25, 51, 391, 450))
        self.TestResult.setObjectName("TestResult")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 1, 1, 1)
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_1 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_1.setGeometry(QtCore.QRect(0, 0, 442, 622))
        self.scrollAreaWidgetContents_1.setObjectName("scrollAreaWidgetContents_1")
        self.GarbageSorting = QtWidgets.QLabel(self.scrollAreaWidgetContents_1)
        self.GarbageSorting.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.GarbageSorting.setFont(font)
        self.GarbageSorting.setObjectName("GarbageSorting")
        self.TestImage = QtWidgets.QLabel(self.scrollAreaWidgetContents_1)
        self.TestImage.setGeometry(QtCore.QRect(20, 90, 401, 481))
        self.TestImage.setObjectName("TestImage")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_1)
        self.gridLayout.addWidget(self.scrollArea_2, 0, 0, 4, 1)
        GarbageSortingUI.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(GarbageSortingUI)
        self.statusbar.setObjectName("statusbar")
        GarbageSortingUI.setStatusBar(self.statusbar)

        self.retranslateUi(GarbageSortingUI)
        QtCore.QMetaObject.connectSlotsByName(GarbageSortingUI)

    def retranslateUi(self, GarbageSortingUI):
        _translate = QtCore.QCoreApplication.translate
        GarbageSortingUI.setWindowTitle(_translate("GarbageSortingUI", "MainWindow"))
        self.Input.setText(_translate("GarbageSortingUI", "打开图片"))
        self.Output.setText(_translate("GarbageSortingUI", "识别图片"))
        self.ShiBieJieGuo.setText(_translate("GarbageSortingUI", "识别结果："))
        self.GarbageSorting.setText(_translate("GarbageSortingUI", "垃圾分类"))
#         self.TestImage.setText(_translate("GarbageSortingUI", "TextLabel"))

