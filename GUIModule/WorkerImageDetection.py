from PyQt5.QtCore import QObject, pyqtSignal, QWaitCondition, QMutex
from PyQt5.QtWidgets import QMessageBox

import detector
import video
from CVModule.DrawImages import ImageHandler


class WorkerImageDetection(QObject):
    error = pyqtSignal(str)
    info = pyqtSignal(str)
    no_detections = pyqtSignal(str)
    finished = pyqtSignal()

    images_ready = pyqtSignal(list)
    batch_info = pyqtSignal(str, int)

    def __init__(self, widget):
        QObject.__init__(self)
        self.widget = widget

    def run(self):
        detector.main(self.widget)


class WorkerVideoDetection(QObject):
    error = pyqtSignal(str)
    info = pyqtSignal(str, int)
    image_ready = pyqtSignal(ImageHandler)
    finished = pyqtSignal()

    def __init__(self, widget):
        QObject.__init__(self)
        self.widget = widget
        self.pause = False
        self.pauseCond = QWaitCondition()
        self.pauseMutex = QMutex()

        self.cancel = False

    def run(self):
        video.main(self.widget)

