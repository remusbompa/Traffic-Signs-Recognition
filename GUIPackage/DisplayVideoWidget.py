import os
import subprocess
from os.path import expanduser

import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import QSize, QDir, QStandardPaths, QFile, QIODevice, QThread, Qt
from PyQt5.QtGui import QKeySequence, QPainter, QGuiApplication, QImageReader, QImageWriter, QPalette, QTextCursor, \
    QPixmap, QColor, QIcon
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
from PyQt5.QtWidgets import QMainWindow, QMenuBar, qApp, QFileDialog, QDialog, QMessageBox, QApplication, QWidget, \
    QSplitter, QVBoxLayout, QLabel, QTextEdit, QScrollArea, QHBoxLayout, QSizePolicy, QProgressBar, QPushButton

from VisionPackage.DrawImages import ImageHandler, ContourHandler
from GUIPackage.WorkerImageDetection import WorkerVideoDetection
from moviepy.editor import VideoFileClip


def adjust_scrollbar(scroll_bar, factor):
    scroll_bar.setValue(int(factor * scroll_bar.value()
                            + ((factor - 1) * scroll_bar.pageStep() / 2)))


def get_duration(filename):
    clip = VideoFileClip(filename)
    return clip.duration


def get_length(filename):
    cap = cv2.VideoCapture(filename)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


class VideoViewer(QMainWindow):

    def __init__(self, title, parent=None):
        super(VideoViewer, self).__init__(parent)
        self.parent = parent
        self.title = title
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.printer = QPrinter()
        self.menu_bar = QMenuBar(self)
        self.firstDialog = True

        self.fileMenu = self.menu_bar.addMenu("&File")
        self.openAct = self.fileMenu.addAction("&Open...", self.open_file)
        self.openAct.setShortcut(QKeySequence.Open)

        self.saveAsAct = self.fileMenu.addAction("&Save As...", self.save_as)
        self.saveAsAct.setEnabled(False)

        self.printAct = self.fileMenu.addAction("&Print...", self.print)
        self.printAct.setShortcut(QKeySequence.Print)
        self.printAct.setEnabled(False)

        self.fileMenu.addSeparator()

        self.exitAct = self.fileMenu.addAction("E&xit", self.close)
        self.exitAct.setShortcut("Ctrl+Q")

        self.editMenu = self.menu_bar.addMenu("&Edit")

        self.copyAct = self.editMenu.addAction("&Copy", self.copy)
        self.copyAct.setShortcut(QKeySequence.Copy)
        self.copyAct.setEnabled(False)

        self.viewMenu = self.menu_bar.addMenu("&View")

        self.zoomInAct = self.viewMenu.addAction("Zoom &In (25%)", self.zoom_in)
        self.zoomInAct.setShortcut(QKeySequence.ZoomIn)
        self.zoomInAct.setEnabled(False)

        self.zoomOutAct = self.viewMenu.addAction("Zoom &Out (25%)", self.zoom_out)
        self.zoomOutAct.setShortcut(QKeySequence.ZoomOut)
        self.zoomOutAct.setEnabled(False)

        self.normalSizeAct = self.viewMenu.addAction("&Normal Size", self.normal_size)
        self.normalSizeAct.setShortcut("Ctrl+S")
        self.normalSizeAct.setEnabled(False)

        self.viewMenu.addSeparator()

        self.fitToWindowAct = self.viewMenu.addAction("&Fit to Window", self.fit_to_window)
        self.fitToWindowAct.setEnabled(True)
        self.fitToWindowAct.setCheckable(True)
        self.fitToWindowAct.setShortcut("Ctrl+F")

        self.helpMenu = self.menu_bar.addMenu("&Help")

        self.helpMenu.addAction("&About", self.about)

        self.wid = DisplayVideoWidget(self)
        self.setCentralWidget(self.wid)

        self.scale_factor = 1.0
        self.update_actions()

        available_size = qApp.desktop().availableGeometry().size()
        width = available_size.width()
        height = available_size.height()
        width *= 0.9
        height *= 0.9
        new_size = QSize(width, height)
        self.resize(new_size)

    def zoom_in(self):
        self.scale_image(1.25)

    def zoom_out(self):
        self.scale_image(0.8)

    def normal_size(self):
        self.wid.image_frame.adjustSize()
        self.scale_factor = 1.0

    def fit_to_window(self):
        fit_to_window = self.fitToWindowAct.isChecked()
        self.wid.scrollArea.setWidgetResizable(fit_to_window)
        if not fit_to_window:
            self.normal_size()
        self.update_actions()

    def scale_image(self, factor):
        self.scale_factor *= factor
        self.wid.image_frame.resize(self.scale_factor * self.wid.image_frame.pixmap().size())
        adjust_scrollbar(self.wid.scrollArea.horizontalScrollBar(), factor)
        adjust_scrollbar(self.wid.scrollArea.verticalScrollBar(), factor)
        self.zoomInAct.setEnabled(self.scale_factor < 3.0)
        self.zoomOutAct.setEnabled(self.scale_factor > 0.333)

    def print(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.wid.image_frame.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.wid.image_frame.pixmap().rect())
            painter.drawPixmap(0, 0, self.wid.image_frame.pixmap())

    def open_file(self):
        dialog = QFileDialog(self, "Open File")
        self.init_image_file_dialog(dialog, QFileDialog.AcceptOpen)
        # if file invalid, try again
        while dialog.exec_() == QDialog.Accepted and not self.load_file(dialog.selectedFiles()[0]):
            message = QMessageBox()
            message.information(self, "Error",
                                "Error loading file. Click ok to retry.")

    def load_file(self, file_name):
        if not os.path.isfile(file_name):
            message = QMessageBox()
            message.information(self, QGuiApplication.applicationDisplayName(),
                                f"Cannot load {QDir.toNativeSeparators(file_name)}: Invalid path {file_name}")
            return False
        new_img = cv2.imread(file_name)
        self.scale_factor = 1.0
        self.wid.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.update_actions()
        self.wid.init_image(new_img)

        if self.fitToWindowAct.isChecked():
            self.wid.image_frame.adjustSize()
        return True

    def update_actions(self):
        self.saveAsAct.setEnabled(True)
        self.copyAct.setEnabled(True)
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def init_image_file_dialog(self, dialog, accept_mode):
        if self.firstDialog:
            self.firstDialog = False
            pictures_locations = QStandardPaths.standardLocations(QStandardPaths.PicturesLocation)
            dialog.setDirectory(QDir.currentPath() if len(pictures_locations) == 0 else pictures_locations[-1])

        mime_type_filters = []
        supported_mime_types = QImageReader.supportedMimeTypes() if (accept_mode == QFileDialog.AcceptOpen) \
            else QImageWriter.supportedMimeTypes()
        for mimeTypeName in supported_mime_types:
            mime_type_filters.append(mimeTypeName.data().decode())
        mime_type_filters.sort()
        dialog.setMimeTypeFilters(mime_type_filters)
        dialog.selectMimeTypeFilter("image/jpeg")
        if accept_mode == QFileDialog.AcceptSave:
            dialog.setDefaultSuffix("jpg")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    def save_as(self):
        qfd = QFileDialog()
        filename, _ = QFileDialog.getSaveFileName(qfd, 'Save image', expanduser("~"), '*.png',
                                                  "", qfd.DontUseNativeDialog)
        file = QFile(filename)
        if not file.open(QIODevice.WriteOnly):
            message = QMessageBox()
            message.information(self, "Unable to open file",
                                file.errorString())
            return False
        self.wid.image_frame.pixmap().save(file, "PNG")

    def copy(self):
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(self.wid.image_frame.pixmap())

    def about(self):
        message = QMessageBox()
        message.about(self, "About TrafficIA",
                      "<p>The <b>Image Viewer</b> example shows how to combine QLabel "
                      "and QScrollArea to display an image. QLabel is typically used "
                      "for displaying a text, but it can also display an image. "
                      "QScrollArea provides a scrolling view around another widget. "
                      "If the child widget exceeds the size of the frame, QScrollArea "
                      "automatically provides scroll bars. </p><p>The example "
                      "demonstrates how QLabel's ability to scale its contents "
                      "(QLabel::scaledContents), and QScrollArea's ability to "
                      "automatically resize its contents "
                      "(QScrollArea::widgetResizable), can be used to implement "
                      "zooming and scaling features. </p><p>In addition the example "
                      "shows how to use QPainter to print an image.</p>")

    def back_to_parent(self):
        self.hide()
        self.parent.show()
        self.centralWidget().deleteLater()
        self.deleteLater()


class DisplayVideoWidget(QWidget):
    def __init__(self, parent: VideoViewer):
        super().__init__()
        self.parent = parent
        self.fitToWindowAct = self.parent.fitToWindowAct
        self.img_handler = None

        layout_h = QSplitter(Qt.Horizontal)
        layout_hv1 = QVBoxLayout()
        self.label1 = QLabel("Processing video infos:")
        self.textEdit1 = QTextEdit()
        self.textEdit1.setReadOnly(True)

        layout_hv1.addWidget(self.label1)
        layout_hv1.addWidget(self.textEdit1)
        topleft = QWidget(self)
        topleft.setLayout(layout_hv1)
        layout_h.addWidget(topleft)

        layout_hv2 = QVBoxLayout()
        self.label2 = QLabel("Select a contour")
        self.textEdit2 = QTextEdit()
        layout_hv2.addWidget(self.label2)
        layout_hv2.addWidget(self.textEdit2)
        topright = QWidget(self)
        topright.setLayout(layout_hv2)
        layout_h.addWidget(topright)

        layout_h.setSizes([500, 100])

        layout_v = QSplitter(Qt.Vertical)
        layout_v.addWidget(layout_h)
        self.image_frame = QLabel()
        self.scrollArea = QScrollArea()
        layout_v.addWidget(self.scrollArea)

        layout_v.setSizes([100, 500])

        layout = QVBoxLayout(self)
        layout.addWidget(layout_v)

        # progress bar for video status
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, self.width(), 25)

        self.pbar.setValue(0)
        layout.addWidget(self.pbar)

        # start and stop buttons for video
        hbox = QHBoxLayout()
        self.start_btn = QPushButton('Start', self)
        self.start_btn.clicked.connect(self.start_process)

        self.cancel_btn = QPushButton('Back', self)
        self.cancel_btn.clicked.connect(self.back_process)

        self.cancel_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

        hbox.addStretch(1)
        hbox.addWidget(self.cancel_btn)
        hbox.addWidget(self.start_btn)

        layout.addLayout(hbox)

        self.setLayout(layout)

        self.args = parent.parent.widget.args
        self.pbar.setMaximum(5 + get_length(self.args.video))
        # manage signals
        self.obj = WorkerVideoDetection(self)
        self.detection_thread = QThread()

        self.obj.moveToThread(self.detection_thread)
        self.detection_thread.started.connect(self.obj.run)

        # Handle closing events
        self.obj.finished.connect(self.finished_process)
        self.obj.finished.connect(self.detection_thread.quit)

        # Handle processing signals from detector
        self.obj.error.connect(self.handle_error)
        self.obj.info.connect(self.handle_info)

        self.obj.image_ready.connect(self.handle_image_ready)
        self.obj.contour_ready.connect(self.show_contour_on_signal)
        # image_frame handling
        self.image_frame.setBackgroundRole(QPalette.Base)
        self.image_frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_frame.setScaledContents(True)
        self.image_frame.mousePressEvent = self.show_contour_info

        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.image_frame)
        self.scrollArea.setVisible(True)

    def finished_process(self):
        QMessageBox.information(self, "Done!", "Done detecting video!")
        self.pbar.setValue(0)
        self.start_btn.setText("Start")
        self.start_btn.clicked.connect(self.start_process)
        self.cancel_btn.setText('Back')
        self.cancel_btn.clicked.connect(self.back_process)

    def handle_error(self, msg):
        message = QMessageBox()
        message.critical(self, "Error", msg)
        message.setFixedSize(500, 200)
        self.parent.hide()
        self.parent.parent.show()
        self.parent.deleteLater()

    def handle_info(self, msg, current_time):
        self.textEdit1.append(msg)
        if current_time == -2:
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)
        elif current_time == -1:
            self.pbar.setValue(0)
        elif current_time == 0:
            self.pbar.setValue(5)
        else:
            self.pbar.setValue(1 + self.pbar.value())

    def handle_image_ready(self, img: ImageHandler):
        self.img_handler = img
        self.init_image(img.imread)

    def start_process(self):
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.start_btn.setText("Stop")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.stop_process)
        self.cancel_btn.setText('Cancel')
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.detection_thread.start()

    def stop_process(self):
        self.start_btn.setText("Resume")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.restart_process)
        self.cancel_btn.setText('Cancel')
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.obj.pause = True

    def restart_process(self):
        self.start_btn.setText("Stop")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.stop_process)
        self.cancel_btn.setText('Cancel')
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.obj.pause = False
        self.obj.pauseCond.wakeAll()

    def cancel_process(self):
        self.pbar.setValue(0)
        self.start_btn.setText('Start')
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.start_process)
        self.cancel_btn.setText('Back')
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.back_process)
        self.obj.cancel = True

    def back_process(self):
        self.parent.back_to_parent()

    def init_image(self, image):
        # Set image in image_frame
        height, width, channels = image.shape
        bytes_per_line = channels * width
        image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(image))

        self.scrollArea.setVisible(True)

        if not self.fitToWindowAct.isChecked():
            self.image_frame.adjustSize()

    def init_image_info(self, img: ImageHandler):
        text_edit = self.textEdit1
        text_edit.append(f"Image name: {img.img_name}")
        text_edit.append(f"Image number: {img.image_ind}")
        text_edit.append(f"Batch number: {img.batch_no}")
        text_edit.append(f"Path: {img.path}")
        text_edit.append("\n" + img.batch_info + "\n")
        contours = img.contours
        headers = ["Number", "Bottom left corner", "Top right corner", "Object confidence",
                   "Class Score", "Class", "Label", "Color"]

        cursor = text_edit.textCursor()
        cursor.insertTable(len(contours) + 1, len(headers))
        for header in headers:
            cursor.insertText(header)
            cursor.movePosition(QTextCursor.NextCell)
        for contour in contours:
            row = [contour.number, contour.corner_bl, contour.corner_tr, contour.obj_conf, contour.cls_score,
                   contour.cls, contour.label, contour.color]
            for value in row:
                cursor.insertText(f"{value}")
                cursor.movePosition(QTextCursor.NextCell)

    def show_contour_info(self, event):
        x = event.pos().x()
        y = event.pos().y()
        qsize = self.image_frame.size()
        x *= self.img_handler.shape[0] / qsize.width()
        y *= self.img_handler.shape[1] / qsize.height()
        text_edit = self.textEdit2

        text_edit.setText(f"x:{x} y:{y}")
        for contour in self.img_handler.contours:
            left, bottom = contour.corner_bl
            right, top = contour.corner_tr

            if left <= x <= right and bottom <= y <= top:
                text_edit.append(f"Contour number: {contour.number}")
                text_edit.append(f"Contour left margin: {left}")
                text_edit.append(f"Contour right margin: {right}")
                text_edit.append(f"Contour bottom margin: {bottom}")
                text_edit.append(f"Contour top margin: {top}")
                text_edit.append(f"Object confidence: {contour.obj_conf}")
                text_edit.append(f"Class score: {contour.cls_score}")
                text_edit.append(f"Label: {contour.label}")
                text_edit.append(f"Tracking ID: {contour.track_id}")
                text_edit.append("Color: ")
                pix_map = QPixmap(20, 20)
                b, g, r = contour.color
                pix_map.fill(QColor(r, g, b))
                color_icon = pix_map.toImage()
                text_cursor = text_edit.textCursor()
                text_cursor.movePosition(
                    QtGui.QTextCursor.NextCell,
                    QtGui.QTextCursor.MoveAnchor
                )
                text_cursor.insertImage(color_icon)
                # This will hide the cursor
                blank_cursor = QtGui.QCursor(Qt.BlankCursor)
                text_edit.setCursor(blank_cursor)
                text_edit.moveCursor(QtGui.QTextCursor.End)
                return

    def show_contour_on_signal(self, contour: ContourHandler):
        text_edit = self.textEdit2
        text_edit.clear()
        left, bottom = contour.corner_bl
        right, top = contour.corner_tr
        text_edit.append(f"Contour number: {contour.number}")
        text_edit.append(f"Contour left margin: {left}")
        text_edit.append(f"Contour right margin: {right}")
        text_edit.append(f"Contour bottom margin: {bottom}")
        text_edit.append(f"Contour top margin: {top}")
        text_edit.append(f"Object confidence: {contour.obj_conf}")
        text_edit.append(f"Class score: {contour.cls_score}")
        text_edit.append(f"Label: {contour.label}")
        text_edit.append(f"Tracking ID: {contour.id_track}")

        text_edit.append("Color: ")
        pix_map = QPixmap(20, 20)
        b, g, r = contour.color
        pix_map.fill(QColor(r, g, b))
        color_icon = pix_map.toImage()
        text_cursor = text_edit.textCursor()
        text_cursor.movePosition(
            QtGui.QTextCursor.NextCell,
            QtGui.QTextCursor.MoveAnchor
        )
        text_cursor.insertImage(color_icon)
        # This will hide the cursor
        blank_cursor = QtGui.QCursor(Qt.BlankCursor)
        text_edit.setCursor(blank_cursor)
        text_edit.moveCursor(QtGui.QTextCursor.End)


def main():
    app = QApplication([])
    image_widget = VideoViewer("Title")
    image_widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
