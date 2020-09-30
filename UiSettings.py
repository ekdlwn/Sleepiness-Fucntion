# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Settings.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

class UiSettings(QtWidgets.QDialog):
    def __init__(self, container, index):
        super(UiSettings, self).__init__(container.container)
        self.container = container
        self.index = index
        self.setupUi()

        parameters = self.container.settings['Camera' + str(index)]
        self.dataFolder.setText(self.container.dataFolder)
        self.fps.setText(str(parameters['fps']))
        self.width.setText(str(parameters['width']))
        self.height.setText(str(parameters['height']))
        self.brightness.setText(str(parameters['brightness']))
        self.contrast.setText(str(parameters['contrast']))
        self.saturation.setText(str(parameters['saturation']))
        self.hue.setText(str(parameters['hue']))
        self.gain.setText(str(parameters['gain']))
        self.exposure.setText(str(parameters['exposure']))

        self.show()

    def setupUi(self):
        self.setObjectName("Settings")
        self.resize(459, 263)
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.label_8 = QtWidgets.QLabel(self)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 4, 1, 2)
        self.hue = QtWidgets.QLineEdit(self)
        self.hue.setObjectName("hue")
        self.gridLayout.addWidget(self.hue, 1, 6, 1, 1)
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 2, 1, 1)
        self.width = QtWidgets.QLineEdit(self)
        self.width.setObjectName("width")
        self.gridLayout.addWidget(self.width, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 2, 1, 1)
        self.contrast = QtWidgets.QLineEdit(self)
        self.contrast.setObjectName("contrast")
        self.gridLayout.addWidget(self.contrast, 2, 3, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 5, 1, 2)
        self.saturation = QtWidgets.QLineEdit(self)
        self.saturation.setObjectName("saturation")
        self.gridLayout.addWidget(self.saturation, 3, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 3, 4, 1, 2)
        self.exposure = QtWidgets.QLineEdit(self)
        self.exposure.setObjectName("exposure")
        self.gridLayout.addWidget(self.exposure, 3, 6, 1, 1)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.height = QtWidgets.QLineEdit(self)
        self.height.setObjectName("height")
        self.gridLayout.addWidget(self.height, 3, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 4, 1, 2)
        self.gain = QtWidgets.QLineEdit(self)
        self.gain.setObjectName("gain")
        self.gridLayout.addWidget(self.gain, 2, 6, 1, 1)
        self.dataFolder = QtWidgets.QLineEdit(self)
        self.dataFolder.setObjectName("filename")
        self.gridLayout.addWidget(self.dataFolder, 0, 1, 1, 4)
        self.brightness = QtWidgets.QLineEdit(self)
        self.brightness.setObjectName("brightness")
        self.gridLayout.addWidget(self.brightness, 1, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.fps = QtWidgets.QLineEdit(self)
        self.fps.setObjectName("fps")
        self.gridLayout.addWidget(self.fps, 1, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Reset)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 3, 1, 4)

        self.pushButton.clicked.connect(self.selectFileDialog)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.close)
        self.buttonBox.button(self.buttonBox.Reset).clicked.connect(self.reset)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Settings", "Settings"))
        self.label_8.setText(_translate("Settings", "Hue"))
        self.label_10.setText(_translate("Settings", "Exposure"))
        self.label_5.setText(_translate("Settings", "Brightness"))
        self.label_3.setText(_translate("Settings", "Width"))
        self.pushButton.setText(_translate("Settings", "Select"))
        self.label_7.setText(_translate("Settings", "Saturation"))
        self.label.setText(_translate("Settings", "Save in"))
        self.label_9.setText(_translate("Settings", "Gain"))
        self.label_6.setText(_translate("Settings", "Contrast"))
        self.label_4.setText(_translate("Settings", "Height"))
        self.label_2.setText(_translate("Settings", "FPS"))

    def selectFileDialog(self):
        options = QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog
        dataFolder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data folder",
                                                                 self.dataFolder.text(), options=options)
        if dataFolder:
            self.dataFolder.setText(dataFolder)

    def accept(self):
        parameters = self.container.settings['Camera' + str(self.index)]
        try:
            parameters['fps'] = float(self.fps.text())
            parameters['width'] = float(self.width.text())
            parameters['height'] = float(self.height.text())
            parameters['brightness'] = float(self.brightness.text())
            parameters['contrast'] = float(self.contrast.text())
            parameters['saturation'] = float(self.saturation.text())
            parameters['hue'] = float(self.hue.text())
            parameters['gain'] = float(self.gain.text())
            parameters['exposure'] = float(self.exposure.text())
            parameters['dataFolder'] = self.dataFolder.text()
            self.container.dataFolder = self.dataFolder.text()
            self.close()
        except:
            pass

    def reset(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return
        self.dataFolder.setText(self.container.dataFolder)
        self.fps.setText(str(cap.get(cv2.CAP_PROP_FPS)))
        self.width.setText(str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height.setText(str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.brightness.setText(str(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
        self.contrast.setText(str(cap.get(cv2.CAP_PROP_CONTRAST)))
        self.saturation.setText(str(cap.get(cv2.CAP_PROP_SATURATION)))
        self.hue.setText(str(cap.get(cv2.CAP_PROP_HUE)))
        self.gain.setText(str(cap.get(cv2.CAP_PROP_GAIN)))
        self.exposure.setText(str(cap.get(cv2.CAP_PROP_EXPOSURE)))
        cap.release()
