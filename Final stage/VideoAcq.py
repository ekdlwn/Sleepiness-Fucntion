# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:59:09 2020

@author: ekdlw
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VideoAcq.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

import os
import sys
import cv2
import json
import uuid
import webbrowser
from subprocess import call
from pylsl import StreamInfo, StreamOutlet

from PyQt5 import QtCore, QtGui, QtWidgets
from Settings import UiSettings


class VideoAcq(object):
    def __init__(self):
        self.defaults = {'fps': 30, 'width': 640, 'height': 480, 'brightness': 0.5, 'contrast': 0.5,
                    'saturation': 0.64, 'hue': 0.5, 'gain': 0.5, 'exposure': float('inf'),
                    'dataFolder': os.path.join(os.path.expanduser('~'), 'Data')}
        self.settings = {}

        if os.path.exists('.preferences.txt'):
            with open('.preferences.txt','r') as hFile:
                try:
                    self.settings = json.load(hFile)
                except:
                    self.settings = self.defaults
        else:
            self.settings = self.defaults
        self.dataFolder = self.settings['dataFolder']
        self.container = None
        self.lock = False
        self.counter = 0
        self.timer = QtCore.QTimer()
        self.timer.timerEvent = self.timerEvent
        self.state = ''
        self.postProcFiles = []

    def setupUi(self, MainWindow):
        self.container = MainWindow
        self.container.closeEvent = self.closeEvent
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(280, 160)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.preview = QtWidgets.QPushButton(self.centralwidget)
        self.preview.setObjectName("preview")
        self.gridLayout.addWidget(self.preview, 0, 1, 1, 1)
        self.help = QtWidgets.QPushButton(self.centralwidget)
        self.help.setObjectName("help")
        self.gridLayout.addWidget(self.help, 4, 1, 1, 1)
        self.rec = QtWidgets.QPushButton(self.centralwidget)
        self.rec.setObjectName("rec")
        self.gridLayout.addWidget(self.rec, 1, 1, 1, 1)
        self.refresh = QtWidgets.QPushButton(self.centralwidget)
        self.refresh.setObjectName("refresh")
        self.gridLayout.addWidget(self.refresh, 3, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tableView = QtWidgets.QTableWidget(self.centralwidget)
        self.tableView.setObjectName("tableView")
        self.horizontalLayout.addWidget(self.tableView)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 5, 1)
        self.stop = QtWidgets.QPushButton(self.centralwidget)
        self.stop.setObjectName("stop")
        self.gridLayout.addWidget(self.stop, 2, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.preview.setIcon(QtGui.QIcon(QtGui.QPixmap('resources/Gnome-media-playback-start.svg')))
        self.rec.setIcon(QtGui.QIcon(QtGui.QPixmap('resources/Gnome-media-record.svg')))
        self.stop.setIcon(QtGui.QIcon(QtGui.QPixmap('resources/Gnome-media-playback-stop.svg')))
        self.refresh.setIcon(QtGui.QIcon(QtGui.QPixmap('resources/Gnome-view-refresh.svg')))
        self.help.setIcon(QtGui.QIcon(QtGui.QPixmap('resources/Gnome-help-browser.svg')))
        self.preview.setToolTip("Play")
        self.rec.setToolTip("Rec")
        self.stop.setToolTip("Stop")
        self.refresh.setToolTip("Refresh")
        self.help.setToolTip("Help")
        self.preview.clicked.connect(self.play)
        self.rec.clicked.connect(self.record)
        self.stop.clicked.connect(self.stopCapture)
        self.refresh.clicked.connect(self.listCapDev)
        self.help.clicked.connect(self.showHelp)
        self.listCapDev()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("VideoAcq", "VideoAcq"))

    def showHelp(self):
        webbrowser.open("https://bitbucket.org/neatlabs/videoacq")

    def timerEvent(self, event):
        if self.counter == 0:
            msg = self.state
        else:
            msg = self.state+'.'*self.counter
        self.counter += 1
        if self.counter > 3:
            self.counter = 0
        self.statusbar.showMessage(msg)

    def listCapDev(self):
        if self.lock:
            return
        self.tableView.clear()
        k = 0
        while True:
            cap = cv2.VideoCapture(k)
            if not cap.isOpened():
                break
            else:
                cap.release()
            k += 1
        n = k

        self.tableView.setColumnCount(2)
        self.tableView.setRowCount(n)
        self.tableView.setHorizontalHeaderLabels(['Camera', 'Settings'])
        iconSettings = QtGui.QIcon(QtGui.QPixmap('resources/Gnome-system-run.svg'))
        for k in range(n):
            chkBoxItem = QtWidgets.QTableWidgetItem()
            chkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(QtCore.Qt.Unchecked)
            self.tableView.setItem(k, 0, chkBoxItem)
            if 'Camera' + str(k) not in self.settings:
                self.settings['Camera' + str(k)] = self.defaults.copy()
            settings = QtWidgets.QPushButton(self.tableView)
            settings.setIcon(iconSettings)
            settings.clicked.connect(self.uiSettings(k))
            self.tableView.setCellWidget(k, 1, settings)

    def play(self):
        self.capture(record=False)

    def record(self):
        self.capture(record=True)

    def capture(self, record = False):
        devIndices = []
        cap = []
        writers = []
        winNames = []
        outlets = []
        frameCounter = 1
        for r in range(self.tableView.rowCount()):
            if self.tableView.item(r, 0).checkState() == QtCore.Qt.Checked:
                devIndices.append(r)
        if len(devIndices) == 0:
            return
        for dev in devIndices:
            cap_i = cv2.VideoCapture(dev)
            if not cap_i.isOpened():
                continue
            cap.append(cap_i)
            setDevParameters(cap_i, self.settings['Camera' + str(dev)])
            winName = 'Camera ' + str(dev)
            winNames.append(winName)
            cv2.namedWindow(winName)

            if record:
                fps = cap_i.get(cv2.CAP_PROP_FPS)
                width = cap_i.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT)
                filename = os.path.join(self.dataFolder, 'Camera' + str(dev) + '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer_i = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
                writers.append(writer_i)
                outlets.append(createOutlet(dev, filename))
                self.postProcFiles.append(filename)

        if record:
            self.state = 'Recording'
        else:
            self.state = 'Capturing'
        self.timer.start(500)
        try:
            ret = True
            while self.state != "Stop" and ret:
                for i, cap_i in enumerate(cap):
                    win_i = winNames[i]
                    if cv2.getWindowProperty(win_i, cv2.WND_PROP_VISIBLE):
                        ret, frame = cap_i.read()
                        cv2.imshow(win_i, frame)
                        if record:
                            outlets[i].push_sample([frameCounter])
                            writers[i].write(frame)
                    else:
                        ret = False
                frameCounter += 1
                cv2.waitKey(1)
        finally:
            for cap_i in cap:
                cap_i.release()
            cv2.destroyAllWindows()
            self.state = ''
            self.timer.stop()
            self.statusbar.showMessage(self.state)

    def stopCapture(self):
        self.state = "Stop"
        #cv2.destroyAllWindows()

    def uiSettings(self, index):
        def uiSettingsCap():
            UiSettings(self, index)
        return uiSettingsCap

    def closeEvent(self, event):
        with open('.preferences.txt', 'w') as hFile:
            json.dump(self.settings, hFile)

        if sys.platform == "linux":
            postProcFiles = list(set(self.postProcFiles))
            n = len(postProcFiles)
            if n==0:
                return
            for k, video in enumerate(postProcFiles):
                if os.path.exists(video):
                    self.statusbar.showMessage('Postprocessing '+str(100*(k+1)/n)+'%')
                    call(['ffmpeg','-i', video, '-c:v', 'libtheora', '-q:v', '7', '-c:a',
                          'libvorbis', '-q:a', '4', os.path.splitext(video)[0]+'.ogv','-y'])
                    call(['rm', video])


def setDevParameters(cap, parameters):
    cap.set(cv2.CAP_PROP_FPS,           parameters['fps'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   parameters['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  parameters['height'])
    cap.set(cv2.CAP_PROP_BRIGHTNESS,    parameters['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST,      parameters['contrast'])
    cap.set(cv2.CAP_PROP_SATURATION,    parameters['saturation'])
    cap.set(cv2.CAP_PROP_HUE,           parameters['hue'])
    cap.set(cv2.CAP_PROP_GAIN,          parameters['gain'])
    cap.set(cv2.CAP_PROP_EXPOSURE,      parameters['exposure'])


def createOutlet(index, filename):
    streamName = 'FrameMarker'+str(index+1)
    info = StreamInfo(name=streamName,
                      type='videostream',
                      channel_format='float32',
                      channel_count=1,
                      source_id=str(uuid.uuid4()))
    if sys.platform == "linux":
        videoFile = os.path.splitext(filename)[0]+'.ogv'
    info.desc().append_child_value("videoFile", filename)
    return StreamOutlet(info)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    ui = VideoAcq()
    ui.setupUi(window)
    window.show()

    # Start the main event loop.
    try:
        app.exec_()
    finally:
        print('Good bye!')
    return 0


if __name__ == "__main__":
    sys.exit(main())
