import argparse
import os
from os.path import expanduser
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication, QThread, Qt, QSize, QFile
from PyQt5.QtGui import QIcon, QTextBlockFormat, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, \
    QMainWindow, QLineEdit, QLabel, QGridLayout, QHBoxLayout, QProgressBar, QMessageBox, QListWidget, qApp

from GUIPackage.ContractQPushButton import ContractQPushButton, ContractQText, SelectQPushButton, SelectQText
from GUIPackage.DisplayImageWidget import ImageViewer
from GUIPackage.VideoGUI import DetectVideoApp
from GUIPackage.WorkerImageDetection import WorkerImageDetection


def images_processed_single_left_click(item):
    item.setSelected(True)


def images_processed_right_click(nb):
    if nb == 1:
        print('Single right click')
    else:
        print('Double right click')


def resize_window(main_win: QMainWindow, percent: float = 0.9):
    available_size = qApp.desktop().availableGeometry().size()
    width = available_size.width()
    height = available_size.height()
    width *= percent
    height *= percent
    new_size = QSize(width, height)
    main_win.resize(new_size)


class ProgressWindow(QWidget):
    def __init__(self, filedialog):
        super().__init__()
        self.filedialog = filedialog
        self.parent = self.filedialog.parent
        self.args = filedialog.args
        self.imageList = None
        self.batchInfos = {}

        self.fmt = QTextBlockFormat()
        self.fmt.setBackground(Qt.red)

        layout = QVBoxLayout()

        self.labelImagesProcessed = QLabel("Images Processed:")
        layout.addWidget(self.labelImagesProcessed)

        self.imagesProcessed = QListWidget()

        self.labels = [os.path.splitext(os.path.basename(img))[0] for img in self.args.images]
        self.imagesProcessed.itemDoubleClicked.connect(self.images_processed_double_left_click)
        self.imagesProcessed.itemClicked.connect(images_processed_single_left_click)
        layout.addWidget(self.imagesProcessed)

        self.labelTextInfos = QLabel("Processed image infos:")
        layout.addWidget(self.labelTextInfos)

        self.textInfos = QTextEdit()
        self.textInfos.setReadOnly(True)
        layout.addWidget(self.textInfos)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, self.width(), 25)
        self.progress_bar.setMaximum(5 + len(self.args.images))
        self.progress_bar.setValue(0)
        self.progress_bar_increase = 0
        layout.addWidget(self.progress_bar)

        horizontal_box = QHBoxLayout()
        self.back_btn = QPushButton('Back', self)
        self.back_btn.clicked.connect(self.back_process)

        self.start_btn = QPushButton('Start', self)
        self.start_btn.clicked.connect(self.start_process)

        self.stop_btn = QPushButton('Stop', self)
        self.stop_btn.clicked.connect(self.stop_process)

        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

        horizontal_box.addWidget(self.back_btn, alignment=Qt.AlignLeft)
        horizontal_box.addStretch(2)
        horizontal_box.addWidget(self.stop_btn)
        horizontal_box.addWidget(self.start_btn)

        layout.addLayout(horizontal_box)
        self.setLayout(layout)

        # define signals
        self.obj = WorkerImageDetection(self)
        self.detection_thread = QThread()

        self.obj.moveToThread(self.detection_thread)
        self.detection_thread.started.connect(self.obj.run)

        # Handle closing events
        self.obj.finished.connect(self.finished_process)
        self.obj.finished.connect(self.detection_thread.quit)
        self.obj.finished.connect(self.obj.deleteLater)
        self.obj.finished.connect(self.detection_thread.deleteLater)

        # Handle processing signals from detector
        self.obj.error.connect(self.handle_error)
        self.obj.info.connect(self.handle_info)
        self.obj.no_detections.connect(self.handle_no_detections)

        self.obj.images_ready.connect(self.handle_images_ready)
        self.obj.batch_info.connect(self.handle_batch_info)

    def start_process(self):
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.detection_thread.start()

    def stop_process(self):
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.detection_thread.terminate()

    def finished_process(self):
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.information(self, "Done!", "Done detecting images!")

    def handle_error(self, msg):
        messagebox = QMessageBox()
        messagebox.critical(self, "Error", msg)
        messagebox.setFixedSize(500, 200)
        self.filedialog.parent.change_widget(self.filedialog)

    def handle_info(self, msg):
        if not self.textInfos:
            self.textInfos.append(msg)
        else:
            self.textInfos.append("\n" + msg)
        if self.progress_bar_increase == 1:
            self.progress_bar.setValue(1 + self.progress_bar.value())
        self.progress_bar_increase = 1 - self.progress_bar_increase

    def handle_batch_info(self, msg, image_id):
        self.batchInfos[image_id] = msg
        self.imagesProcessed.addItem(self.labels[image_id])

    def handle_no_detections(self, msg):
        message = QMessageBox()
        message.warning(self, "Warning", msg)
        message.setFixedSize(500, 200)

    def handle_images_ready(self, image_list):
        self.imageList = image_list
        for image in image_list:
            image.set_batch_info(self.batchInfos[image.image_ind])

    def images_processed_double_left_click(self):
        current_id = self.imagesProcessed.currentRow()
        if self.imageList:
            img = self.imageList[current_id]
            self.parent.show_image_viewer(img, self.tr(f"{self.labels[current_id].capitalize()} Detection View"))
        else:
            QMessageBox.information(self, f"{self.labels[current_id]} detection info", self.batchInfos[current_id])

    def back_process(self):
        self.parent.back_to_parent()
        self.deleteLater()


def cancel_detection():
    QCoreApplication.quit()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Traffic IA'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('../images/Logo.png'))

        style_file = "../Style/main.qss"
        with open(style_file, "r") as sf:
            self.setStyleSheet(sf.read())

        widget = ChooseActionWidget(self)
        self.setCentralWidget(widget)

        resize_window(self)

    def start_detect_images(self):
        self.hide()
        images_win = DetectImagesApp(self)
        images_win.show()

    def start_detect_video(self):
        self.hide()
        video_win = DetectVideoApp(self)
        video_win.show()

    def start_image_statistics(self):
        pass


class ChooseActionWidget(QWidget):
    def __init__(self, parent: App):
        super().__init__()
        layout = QVBoxLayout()
        self.logo_image = QLabel(self)
        pix_map = QPixmap("../images/white_Logo.png")
        self.logo_image.setPixmap(pix_map)
        # self.logo_image.setFixedSize(15, 15)
        self.name_app = QLabel("Traffic IA")
        self.name_app.setFont(QFont("Arial", 60, QFont.Bold))
        layout_horizontal = QHBoxLayout()
        layout_horizontal.addWidget(self.logo_image, alignment=Qt.AlignRight)
        layout_horizontal.addWidget(self.name_app, alignment=Qt.AlignLeft)
        self.detect_image_button = QPushButton("Detect images")
        self.detect_image_button.clicked.connect(parent.start_detect_images)
        layout.addLayout(layout_horizontal)
        layout.addStretch(1)
        layout.addLayout(ContractQPushButton(self.detect_image_button))
        layout.addStretch(1)
        self.detect_video_button = QPushButton("Detect in video")
        self.detect_video_button.clicked.connect(parent.start_detect_video)
        layout.addLayout(ContractQPushButton(self.detect_video_button))
        layout.addStretch(1)
        self.image_statistics_button = QPushButton("Detection statistics")
        self.image_statistics_button.clicked.connect(parent.start_image_statistics)
        layout.addLayout(ContractQPushButton(self.image_statistics_button))
        layout.addStretch(1)

        self.setLayout(layout)


class DetectImagesApp(QMainWindow):

    def __init__(self, parent: App = None):
        super(DetectImagesApp, self).__init__(parent)
        self.title = 'Detect traffic signs in images'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('../images/Logo.png'))
        widget = FileDialog(self)
        self.setCentralWidget(widget)

        resize_window(self)

    def change_widget(self, widget):
        self.setCentralWidget(widget)
        self.show()

    def back_to_parent(self):
        self.hide()
        self.parent().show()
        self.centralWidget().deleteLater()
        self.deleteLater()

    def show_image_viewer(self, img, title):
        image_widget = ImageViewer(img, title, self)
        image_widget.show()


class FileDialog(QWidget):
    def __init__(self, parent: DetectImagesApp = None):
        super().__init__()
        self.parent = parent
        self.args = argparse.Namespace()
        self.setWindowIcon(QIcon('../images/Logo.png'))
        layout = QVBoxLayout()

        self.startDir = os.getcwd()
        # Create images selector
        self.btnIm = QPushButton("Select images to detect")
        self.btnIm.clicked.connect(self.get_images)
        self.textIm = QTextEdit()
        self.textIm.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnIm))
        layout.addLayout(SelectQText(self.textIm))
        layout.addStretch(1)
        # Select destination folder
        self.btnDest = QPushButton("Select destination folder")
        self.btnDest.clicked.connect(self.get_destination)
        self.textDest = QLineEdit()
        self.textDest.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnDest))
        layout.addLayout(SelectQText(self.textDest))
        layout.addStretch(1)
        # Select weights file
        self.btnW = QPushButton("Select weights file")
        self.btnW.clicked.connect(self.get_weights)
        self.textW = QLineEdit()
        self.textW.setReadOnly(True)
        layout.addLayout(SelectQPushButton(self.btnW))
        layout.addLayout(SelectQText(self.textW))
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
        self.resoEdit = QLineEdit()

        self.bsEdit.setText("1")
        self.confEdit.setText("0.5")
        self.nmsEdit.setText("0.4")
        self.resoEdit.setText("416")

        self.textIm.setText("../S1.jpg")
        self.args.images = ["../S1.jpg"]
        self.textDest.setText("../det")
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
        grid.addWidget(self.resoEdit, 4, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)

        layout.addLayout(grid)
        layout.addStretch(1)

        back_button = QPushButton("Back")
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        horizontal_box = QHBoxLayout()
        horizontal_box.addWidget(back_button, alignment=Qt.AlignLeft)
        horizontal_box.addStretch(2)
        horizontal_box.addWidget(ok_button)
        horizontal_box.addWidget(cancel_button)
        back_button.clicked.connect(self.back_detection)
        ok_button.clicked.connect(self.start_detection)
        cancel_button.clicked.connect(cancel_detection)
        layout.addLayout(horizontal_box)

        self.setLayout(layout)

    def get_images(self):
        file_names = QFileDialog.getOpenFileNames(self, 'Open Images', self.startDir,
                                                  'Images (*.png *.xpm *.jpg *.jpeg *.pmp)',
                                                  "", QFileDialog.DontUseNativeDialog)
        if file_names[0]:
            self.args.images = file_names[0]
            self.textIm.setText('\n'.join(file_names[0]))
            self.startDir = str(Path(file_names[0][0]).parent)

    def get_destination(self):
        destination_dir = QFileDialog.getExistingDirectory(self, "Open Directory",
                                                           self.startDir, QFileDialog.ShowDirsOnly)
        if destination_dir:
            self.args.det = destination_dir
            self.textDest.setText(destination_dir)
            self.startDir = destination_dir

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
        filename = self.getOpenFileName(self, 'Open Namesfile', self.startDir, 'Nmesfile (*.names)',
                                               "", QFileDialog.DontUseNativeDialog)
        if filename[0]:
            self.args.names = filename[0]
            self.textNames.setText(filename[0])
            self.startDir = str(Path(filename[0]).parent)

    def start_detection(self):
        self.args.bs = self.bsEdit.text()
        self.args.confidence = self.confEdit.text()
        self.args.nms_thresh = self.nmsEdit.text()
        self.args.reso = self.resoEdit.text()

        for arg in vars(self.args):
            if not arg:
                QMessageBox.critical(self, "Error!", f"Parameter {arg} is empty!")

        progress_window = ProgressWindow(self)
        self.parent.change_widget(progress_window)

    def back_detection(self):
        self.parent.back_to_parent()


def main():
    app = QApplication([])
    ex = App()
    ex.show()
    app.exec_()


if __name__ == "__main__":
    main()
