import argparse
import os
from os.path import expanduser
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QGridLayout, \
    QHBoxLayout, QFileDialog, QCheckBox, QMessageBox, qApp

from GUIPackage.ContractQPushButton import SelectQPushButton, SelectQText
from GUIPackage.DisplayVideoWidget import VideoViewer


def resize_window(main_win: QMainWindow, percent: float = 0.9):
    available_size = qApp.desktop().availableGeometry().size()
    width = available_size.width()
    height = available_size.height()
    width *= percent
    height *= percent
    new_size = QSize(width, height)
    main_win.resize(new_size)


class DetectVideoApp(QMainWindow):

    def __init__(self, parent=None):
        super(DetectVideoApp, self).__init__(parent)
        self.title = 'Detect traffic signs in video'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('../images/Logo.png'))
        self.widget = FileDialog(self)
        self.setCentralWidget(self.widget)

        resize_window(self)

    def change_widget(self, widget):
        self.centralWidget().hide()
        self.centralWidget().deleteLater()
        self.setCentralWidget(widget)
        self.show()

    def show_video_viewer(self, title):
        self.hide()
        video_win = VideoViewer(title, self)
        video_win.show()

    def back_to_parent(self):
        self.hide()
        self.parent().show()
        self.centralWidget().deleteLater()
        self.deleteLater()


def cancel_detection():
    QCoreApplication.quit()


class FileDialog(QWidget):
    def __init__(self, parent: DetectVideoApp = None):
        super().__init__()
        self.parent = parent
        self.args = argparse.Namespace()
        self.setWindowIcon(QIcon('../images/Logo.png'))
        layout = QVBoxLayout()

        self.startDir = os.getcwd()
        # Create video selector
        self.useWebCam = QCheckBox("Use web cam")
        self.useWebCam.setChecked(False)
        self.useWebCamLabel = QLabel("Use web cam")
        self.useWebCam.stateChanged.connect(self.use_webcam_clicked)
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.useWebCamLabel)
        horizontal_layout.addWidget(self.useWebCam, alignment=Qt.AlignLeft)
        horizontal_layout.addStretch(1)
        self.btnIm = QPushButton("Select video to detect")
        self.btnIm.clicked.connect(self.get_video)
        self.textIm = QTextEdit()
        layout.addLayout(horizontal_layout)
        layout.addStretch(1)
        layout.addLayout(SelectQPushButton(self.btnIm))
        layout.addLayout(SelectQText(self.textIm))
        layout.addStretch(1)
        self.textIm.setReadOnly(True)
        # Select destination folder
        self.btn_det = QPushButton("Select destination folder")
        self.btn_det.clicked.connect(self.get_destination)
        self.text_det = QLineEdit()
        self.text_det.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btn_det))
        layout.addLayout(SelectQText(self.text_det))
        layout.addStretch(1)
        # Select weights file
        self.btnW = QPushButton("Select weights file")
        self.btnW.clicked.connect(self.get_weights)
        self.textW = QLineEdit()
        self.textW.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnW))
        layout.addLayout(SelectQPushButton(self.textW))
        layout.addStretch(1)
        # Select Config file
        self.btnConf = QPushButton("Select Config file")
        self.btnConf.clicked.connect(self.get_config)
        self.textConf = QLineEdit()
        self.textConf.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnConf))
        layout.addLayout(SelectQText(self.textConf))
        layout.addStretch(1)
        # Select Names file
        self.btnNames = QPushButton("Select Names file")
        self.btnNames.clicked.connect(self.get_names)
        self.textNames = QLineEdit()
        self.textNames.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnNames))
        layout.addLayout(SelectQText(self.textNames))
        layout.addStretch(1)
        bs_label = QLabel('Batch size')
        conf_label = QLabel('Confidence')
        nms_label = QLabel('Nms threshold')
        res_label = QLabel('Resolution')

        self.bsEdit = QLineEdit()
        self.confEdit = QLineEdit()
        self.nmsEdit = QLineEdit()
        self.resEdit = QLineEdit()

        self.bsEdit.setText("1")
        self.confEdit.setText("0.5")
        self.nmsEdit.setText("0.4")
        self.resEdit.setText("416")

        self.textIm.setText("../driving_Sweden.mp4")
        self.args.video = "../driving_Sweden.mp4"
        self.text_det.setText("../det")
        self.args.det = "../det"
        self.textW.setText("../weights/Swedish.weights")
        self.args.weights = "../weights/Swedish.weights"
        self.textConf.setText("../cfg/Swedish.cfg")
        self.args.cfg = "../cfg/Swedish.cfg"
        self.textNames.setText("../data/Swedish.names")
        self.args.names = "../data/Swedish.names"

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(bs_label, 1, 0)
        grid.addWidget(conf_label, 2, 0)
        grid.addWidget(nms_label, 3, 0)
        grid.addWidget(res_label, 4, 0)

        grid.addWidget(self.bsEdit, 1, 1)
        grid.addWidget(self.confEdit, 2, 1)
        grid.addWidget(self.nmsEdit, 3, 1)
        grid.addWidget(self.resEdit, 4, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)

        layout.addLayout(grid)
        layout.addStretch(1)

        back_button = QPushButton("Back")
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        hor_box = QHBoxLayout()
        hor_box.addWidget(back_button, alignment=Qt.AlignLeft)
        hor_box.addStretch(2)
        hor_box.addWidget(ok_button)
        hor_box.addWidget(cancel_button)
        back_button.clicked.connect(self.back_detection)
        ok_button.clicked.connect(self.start_detection)
        cancel_button.clicked.connect(cancel_detection)
        layout.addLayout(hor_box)

        self.setLayout(layout)

    def get_video(self):
        file_names = QFileDialog.getOpenFileName(self, 'Open Video', self.startDir,
                                                  'Videos (*.webm *.mpg *.ogg *.mp4 *.avi *.mov)', "",
                                                  QFileDialog.DontUseNativeDialog)
        if file_names[0]:
            self.args.video = file_names[0]
            self.textIm.setText(file_names[0])
            self.startDir = str(Path(file_names[0][0]).parent)

    def use_webcam_clicked(self, checked):
        if checked:
            self.args.video = 0
            self.textIm.setText("")
            self.textIm.setReadOnly(True)
            self.btnIm.clicked.disconnect()
            self.btnIm.hide()
            self.textIm.hide()
        else:
            self.args.video = None
            self.textIm.setReadOnly(False)
            self.btnIm.clicked.connect(self.get_video)
            self.btnIm.show()
            self.textIm.show()

    def get_destination(self):
        dir_dest = QFileDialog.getExistingDirectory(self, "Open Directory",
                                                    self.startDir, QFileDialog.ShowDirsOnly)
        if dir_dest:
            self.args.det = dir_dest
            self.text_det.setText(dir_dest)
            self.startDir = dir_dest

    def get_weights(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Weightsfile', self.startDir, 'Weightsfile (*.weights)',
                                               "", QFileDialog.DontUseNativeDialog)
        if filename[0]:
            self.args.weights = filename[0]
            self.textW.setText(filename[0])
            self.startDir = str(Path(filename[0]).parent)

    def get_config(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Configfile', self.startDir, 'Configfile (*.cfg)',
                                               "", QFileDialog.DontUseNativeDialog)
        if filename[0]:
            self.args.cfg = filename[0]
            self.textConf.setText(filename[0])
            self.startDir = str(Path(filename[0]).parent)

    def get_names(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Namesfile', self.startDir, 'Nmesfile (*.names)',
                                               "", QFileDialog.DontUseNativeDialog)
        if filename[0]:
            self.args.names = filename[0]
            self.textNames.setText(filename[0])
            self.startDir = str(Path(filename[0]).parent)

    def start_detection(self):
        self.args.bs = self.bsEdit.text()
        self.args.confidence = self.confEdit.text()
        self.args.nms_thresh = self.nmsEdit.text()
        self.args.reso = self.resEdit.text()

        for arg in vars(self.args):
            if not arg:
                QMessageBox.critical(self, "Error!", f"Parameter {arg} is empty!")

        self.parent.show_video_viewer("Video Viewer")

    def back_detection(self):
        self.parent.back_to_parent()
