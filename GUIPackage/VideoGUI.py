import argparse
import os
from os.path import expanduser
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QGridLayout, \
    QHBoxLayout, QFileDialog, QCheckBox, QMessageBox, qApp, QComboBox, QStyledItemDelegate

from GUIPackage.ContractQPushButton import SelectQPushButton, SelectQText, SelectQCombo
from GUIPackage.DisplayVideoWidget import VideoViewer
from StoragePackage import DatasetsManager
from StoragePackage.DatasetsManager import DataSetsManager


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
        self.args.is_classifier = False
        self.classifier_cfg = None
        self.classifier_weights = None
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
        layout.addLayout(horizontal_layout, 1)
        layout.addStretch(1)
        layout.addLayout(SelectQPushButton(self.btnIm), 1)
        layout.addLayout(SelectQText(self.textIm), 2)
        layout.addStretch(1)
        self.textIm.setReadOnly(True)

        # Select data set
        self.select_ds_label = QLabel("Select dataset")
        self.select_ds = DataSetsManager.get_data_set_combo()
        self.select_ds.setObjectName("SelectCombo")
        self.select_ds.currentTextChanged.connect(self.on_data_set_changed)
        layout.addLayout(SelectQCombo(self.select_ds_label, self.select_ds), 2)
        layout.addStretch(1)
        # Select weights file
        self.btnW = QPushButton("Select weights file")
        self.btnW.clicked.connect(self.get_weights)
        self.textW = QLineEdit()
        self.textW.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnW), 1)
        layout.addLayout(SelectQPushButton(self.textW), 1)
        layout.addStretch(1)
        # Select Config file
        self.btnConf = QPushButton("Select Config file")
        self.btnConf.clicked.connect(self.get_config)
        self.textConf = QLineEdit()
        self.textConf.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnConf), 1)
        layout.addLayout(SelectQText(self.textConf), 1)
        layout.addStretch(1)
        # Select Names file
        self.btnNames = QPushButton("Select Names file")
        self.btnNames.clicked.connect(self.get_names)
        self.textNames = QLineEdit()
        self.textNames.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnNames), 1)
        layout.addLayout(SelectQText(self.textNames), 1)
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

        self.textIm.setText("../vid1_Driving_in_Gothenburg_Sweden.mp4")
        self.args.video = "../vid1_Driving_in_Gothenburg_Sweden.mp4"
        self.on_data_set_changed('Swedish')
        self.select_ds.setCurrentText('Swedish')

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

        layout.addLayout(grid, 5)

        tracking_layout = QHBoxLayout()
        self.use_tracking = QCheckBox("Use tracking")
        self.use_tracking.setChecked(True)
        self.use_tracking_label = QLabel("Use tracking: ")
        self.use_tracking.stateChanged.connect(self.use_tracking_clicked)
        tracking_layout.addWidget(self.use_tracking_label)
        tracking_layout.addWidget(self.use_tracking)
        tracking_layout.addStretch(1)
        layout.addLayout(tracking_layout, 1)

        self.tracking = None
        self.select_tracking_label = QLabel("Select tracking")
        self.select_tracking = QComboBox()
        self.select_tracking.setItemDelegate(QStyledItemDelegate())
        self.select_tracking.setObjectName("SelectCombo")
        self.select_tracking.addItems(["Sort", "Deep Sort"])
        self.select_tracking.currentIndexChanged.connect(self.selection_tracking_change)
        self.select_tracking.setCurrentIndex(1)
        layout.addLayout(SelectQCombo(self.select_tracking_label, self.select_tracking), 2)

        count_layout = QHBoxLayout()
        self.count_enabled = False
        self.use_count = QCheckBox("Count performance")
        self.use_count.setChecked(False)
        self.use_count_label = QLabel("Count statistics: ")
        count_layout.addWidget(self.use_count_label)
        count_layout.addWidget(self.use_count)
        count_layout.addStretch(1)
        layout.addLayout(count_layout, 1)

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
        layout.addLayout(hor_box, 2)

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

    def selection_tracking_change(self, i):
        if i == 0:
            self.tracking = "sort"
        else:
            self.tracking = "deep_sort"

    def use_tracking_clicked(self, checked):
        if checked:
            self.select_tracking.show()
            self.select_tracking.itemText(1)
            self.select_tracking_label.show()
            self.tracking = "deep_sort"
        else:
            self.select_tracking.hide()
            self.select_tracking_label.hide()
            self.tracking = None

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

        self.count_enabled = self.use_count.isChecked()
        if self.select_ds.currentText() in ["European", "Belgium", "GTSRB"]:
            self.args.is_classifier = True
            base_model = "Croat"
            self.set_classifier_parameters(base_model)
        self.parent.show_video_viewer("Video Viewer")

    def set_classifier_parameters(self, set_name):
        self.args.classifier_cfg = f"../cfg/{set_name}.cfg"
        self.args.classifier_weights = f"../weights/{set_name}.weights"
        self.args.classifier_names = f"../data/{set_name}.names"
        self.args.classifier_inp_dim = 416
        self.args.classifier_confidence = 0.5
        self.args.classifier_nms_thresh = 0.4

    def back_detection(self):
        self.parent.back_to_parent()

    def on_data_set_changed(self, text):
        self.args.weights = f"../weights/{text}.weights"
        self.textW.setText(self.args.weights)

        self.args.cfg = f"../cfg/{text}.cfg"
        self.textConf.setText(self.args.cfg)

        self.args.names = f"../data/{text}.names"
        self.textNames.setText(self.args.names)
