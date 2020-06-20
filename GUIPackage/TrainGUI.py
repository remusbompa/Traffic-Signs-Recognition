from PyQt5.QtCore import QCoreApplication, Qt, QProcess, QUrl, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont, QStandardItemModel, QStandardItem, QTextDocument, QPalette, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel, QComboBox, \
    QTextEdit, QWidget, QCheckBox, QSizePolicy, qApp
from StoragePackage.DatasetsManager import DataSetsManager


def resize_window(main_win: QMainWindow, percent: float = 0.9):
    available_size = qApp.desktop().availableGeometry().size()
    width = available_size.width()
    height = available_size.height()
    width *= percent
    height *= percent
    new_size = QSize(width, height)
    main_win.resize(new_size)


class TrainGUI(QMainWindow):
    def __init__(self, parent=None):
        super(TrainGUI, self).__init__(parent)
        self.title = 'Train Traffic IA'
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

    def back_to_parent(self):
        self.hide()
        self.parent().show()
        self.centralWidget().deleteLater()
        self.deleteLater()


def cancel_detection():
    QCoreApplication.quit()


class FileDialog(QWidget):
    def __init__(self, parent: TrainGUI = None):
        super().__init__()
        self.parent = parent
        self.setWindowIcon(QIcon('../images/Logo.png'))
        layout = QVBoxLayout()
        # first line: logo and function name
        self.logo_image = QLabel(self)
        self.w = 3 * self.logo_image.width()
        self.h = 3 * self.logo_image.height()
        """
        pix_map = QPixmap("../images/white_Logo.png")
        self.logo_image.setPixmap(pix_map.scaled(w,h,Qt.KeepAspectRatio))
        self.name_app = QLabel("Traffic IA")
        self.name_app.setFont(QFont("Arial", 30, QFont.Bold))
        layout_horizontal1 = QHBoxLayout()
        layout_horizontal1.addWidget(self.logo_image, alignment=Qt.AlignRight)
        layout_horizontal1.addWidget(self.name_app, alignment=Qt.AlignLeft)

        layout.addLayout(layout_horizontal1, 2)
        layout.addStretch(1)
        """
        # second line: text and list of models to choose from
        layout_horizontal2 = QHBoxLayout()
        self.select_model_label = QLabel('Select model to train: ')
        self.select_model = DataSetsManager.get_data_set_combo()
        self.ds = DataSetsManager.get_data_set(self.select_model.currentText())
        self.select_model.currentTextChanged.connect(self.data_set_changed)
        layout_horizontal2.addWidget(self.select_model_label)
        layout_horizontal2.addWidget(self.select_model, alignment=Qt.AlignLeft)
        layout_horizontal2.addStretch(2)

        layout.addLayout(layout_horizontal2, 5)
        # third line: html description of the data set or terminal output
        self.show_training_info = QTextEdit()
        self.show_training_info.setReadOnly(True)
        self.show_training_info.setStyleSheet("QTextEdit { background-color: rgba(255, 255, 255, 200) }")
        p = self.show_training_info.palette()
        p.setColor(QPalette.Text, Qt.black)
        self.show_training_info.setPalette(p)
        layout.addWidget(self.show_training_info, 8)
        self.show_training_info.clear()
        self.show_training_info.insertHtml(self.ds.html_text)
        self.show_training_info.verticalScrollBar().setValue(0)
        self.show_training_info.moveCursor(QTextCursor.Start)
        layout.addWidget(QLabel("GPU infos:"))
        # fourth line: gpu info
        self.gpu_util_buffer = []
        self.gpu_info = QTextEdit()
        self.gpu_info.setStyleSheet("QTextEdit { background-color: rgba(255, 255, 255, 200) }")
        self.process_gpu_info = QProcess(self)
        self.process_gpu_info.setProgram("/bin/sh")
        ssh_args = ["-c", "sshpass -p stud_remus ssh remus@141.85.232.73 nvidia-smi -l 1"]
        self.process_gpu_info.setProcessChannelMode(QProcess.MergedChannels)
        self.process_gpu_info.setArguments(ssh_args)
        self.process_gpu_info.readyReadStandardOutput.connect(self.on_output_gpu_info)
        self.process_gpu_info.finished.connect(self.on_finished_gpu_info)
        self.process_gpu_info.start()

        self.first_time = True
        self.gpu0_used = False
        self.gpu1_used = False
        self.process_gpu_utilization = QProcess(self)
        self.process_gpu_utilization.setProgram("/bin/sh")
        ssh_args = ["-c", "sshpass -p stud_remus ssh remus@141.85.232."
                          "73 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1"]
        self.process_gpu_utilization.setProcessChannelMode(QProcess.MergedChannels)
        self.process_gpu_utilization.setArguments(ssh_args)
        self.process_gpu_utilization.readyReadStandardOutput.connect(self.on_output_gpu_util)
        self.process_gpu_utilization.finished.connect(self.on_finished_gpu_util)
        self.process_gpu_utilization.start()

        layout.addWidget(self.gpu_info, 5)
        # fifth line: select gpu ans start/stop, back buttons
        self.state_icon0 = QLabel()
        self.state_icon0.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
        self.gpu_icon0 = QLabel()
        self.gpu_icon0.setPixmap(QPixmap("../icons/gpu.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
        self.state_icon1 = QLabel()
        self.state_icon1.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
        self.gpu_icon1 = QLabel()
        self.gpu_icon1.setPixmap(QPixmap("../icons/gpu.png").scaled(self.w, self.h, Qt.KeepAspectRatio))

        select_gpus_layout = QHBoxLayout()
        # GPU0
        self.gpu0_layout = QHBoxLayout()
        self.gpu0_layout_h1 = QVBoxLayout()
        self.check_gpu0 = QCheckBox()
        self.gpu0_layout_h1.addWidget(self.state_icon0)
        self.gpu0_layout_h1.addWidget(self.check_gpu0)
        self.gpu0_layout_h2 = QVBoxLayout()
        self.gpu0_layout_h2.addWidget(self.gpu_icon0)
        self.gpu0_text = QLabel("GPU 0 is available: ??%")
        self.gpu0_layout_h2.addWidget(self.gpu0_text)
        self.gpu0_layout.addLayout(self.gpu0_layout_h2)
        self.gpu0_layout.addLayout(self.gpu0_layout_h1)

        # GPU1
        self.gpu1_layout = QHBoxLayout()
        self.gpu1_layout_h1 = QVBoxLayout()
        self.check_gpu1 = QCheckBox()
        self.gpu1_layout_h1.addWidget(self.state_icon1)
        self.gpu1_layout_h1.addWidget(self.check_gpu1)
        self.gpu1_layout_h2 = QVBoxLayout()
        self.gpu1_layout_h2.addWidget(self.gpu_icon1)
        self.gpu1_text = QLabel("GPU 1 is available: ??%")
        self.gpu1_layout_h2.addWidget(self.gpu1_text)
        self.gpu1_layout.addLayout(self.gpu1_layout_h2)
        self.gpu1_layout.addLayout(self.gpu1_layout_h1)

        select_gpus_layout.addLayout(self.gpu0_layout, 1)
        select_gpus_layout.addStretch(5)
        select_gpus_layout.addLayout(self.gpu1_layout, 1)

        self.process_train = QProcess(self)
        self.process_train.setProgram("/bin/sh")
        self.process_train.setProcessChannelMode(QProcess.MergedChannels)

        self.back_button = QPushButton("Back")
        self.start_button = QPushButton("Start")
        hor_box = QHBoxLayout()
        hor_box.addLayout(select_gpus_layout, 7)
        hor_box.addStretch(2)
        hor_box.addWidget(self.back_button, 1)
        hor_box.addWidget(self.start_button, 1)
        self.back_button.clicked.connect(self.back_train)
        self.start_button.clicked.connect(self.start_train)

        layout.addLayout(hor_box, 1)

        self.setLayout(layout)

    def data_set_changed(self, text):
        self.ds = DataSetsManager.get_data_set(text)
        self.show_training_info.clear()
        self.show_training_info.insertHtml(self.ds.html_text)
        self.show_training_info.verticalScrollBar().setValue(0)

    def on_output_gpu_info(self):
        text = self.process_gpu_info.readAllStandardOutput().data().decode()
        if text.startswith(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            self.gpu_info.clear()
        self.gpu_info.append(text)
        self.gpu_info.verticalScrollBar().setValue(0)

    def on_finished_gpu_info(self):
        self.gpu_info.clear()
        text = self.process_gpu_info.readAllStandardOutput().data().decode()
        self.gpu_info.append(text)
        self.gpu_info.verticalScrollBar().setValue(0)

    def on_output_gpu_util(self):
        text = self.process_gpu_utilization.readAllStandardOutput().data().decode()
        gpu_utils = list(filter(None, text.split("\n")))
        self.gpu_util_buffer.extend(gpu_utils)
        if len(self.gpu_util_buffer) == 2:
            gpu_0 = int(self.gpu_util_buffer[0][:-1])
            gpu_1 = int(self.gpu_util_buffer[1][:-1])
            self.gpu_util_buffer.clear()
            if gpu_0 == 0:
                self.gpu0_text.setText(f"GPU 0 is available: {gpu_0} % used")
            else:
                self.gpu0_text.setText(f"GPU 0 is busy: {gpu_0} % used")

            if gpu_1 == 0:
                self.gpu1_text.setText(f"GPU 1 is available: {gpu_1} % used")
            else:
                self.gpu1_text.setText(f"GPU 1 is busy: {gpu_1} % used")

            if self.first_time:
                self.first_time = False
                if gpu_0 == 0:
                    self.state_icon0.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h,
                                                                                            Qt.KeepAspectRatio))
                    self.gpu0_used = False
                else:
                    self.state_icon0.setPixmap(QPixmap("../icons/busy.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
                    self.gpu0_used = True

                if gpu_1 == 0:
                    self.state_icon1.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h,
                                                                                            Qt.KeepAspectRatio))
                    self.gpu1_used = False
                else:
                    self.state_icon1.setPixmap(QPixmap("../icons/busy.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
                    self.gpu1_used = True
            else:
                if self.gpu0_used:
                    if gpu_0 == 0:
                        self.state_icon0.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h,
                                                                                                Qt.KeepAspectRatio))
                        self.gpu0_used = False
                else:
                    if gpu_0 > 0:
                        self.state_icon0.setPixmap(QPixmap("../icons/busy.png").scaled(self.w, self.h,
                                                                                       Qt.KeepAspectRatio))
                        self.gpu0_used = True

                if self.gpu1_used:
                    if gpu_1 == 0:
                        self.state_icon1.setPixmap(QPixmap("../icons/available.png").scaled(self.w, self.h,
                                                                                                Qt.KeepAspectRatio))
                        self.gpu1_used = False
                else:
                    if gpu_1 > 0:
                        self.state_icon1.setPixmap(QPixmap("../icons/busy.png").scaled(self.w, self.h,
                                                                                       Qt.KeepAspectRatio))
                        self.gpu1_used = True

    def on_finished_gpu_util(self):
        pass

    def start_train(self):
        gpus = []
        if self.check_gpu0.isChecked() and self.check_gpu1.isChecked():
            gpus = ["-gpus", "0,1"]
        elif self.check_gpu0.isChecked():
            gpus = ["-gpus", "0"]
        elif self.check_gpu1.isChecked():
            gpus = ["-gpus", "1"]

        self.start_button.setText("Stop")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self.stop_train)
        self.select_model.setEnabled(False)
        self.check_gpu0.setCheckable(False)
        self.check_gpu1.setCheckable(False)

        command = " ".join(["./darknet", "detector", "train"]
                           + self.ds.parameters + ["-map", "-dont_show"] + gpus)
        ssh_arg = ["-c", f"sshpass -p stud_remus ssh remus@141.85.232.73 ''cd /opt/remus/darknet_European/; {command}'"]
        self.process_train.setArguments(ssh_arg)
        self.process_train.readyReadStandardOutput.connect(self.on_train_ready)
        self.process_train.finished.connect(self.on_train_finished)
        self.show_training_info.clear()
        self.process_train.start()

    def stop_train(self):
        self.process_train.close()
        kill_on_server = QProcess()
        kill_on_server.setProgram("/bin/sh")
        kill_on_server.setArguments(["-c", "sshpass -p stud_remus ssh remus@141.85.232.73 pkill darknet"])
        kill_on_server.start()
        kill_on_server.waitForFinished()
        self.start_button.setText("Start")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self.start_train)
        self.select_model.setEnabled(True)
        self.check_gpu0.setCheckable(True)
        self.check_gpu1.setCheckable(True)
        self.check_gpu0.setChecked(False)
        self.check_gpu1.setChecked(False)

    def on_train_ready(self):
        text = self.process_train.readAllStandardOutput().data().decode()
        self.show_training_info.append(text)

    def on_train_finished(self):
        self.start_button.setText("Start")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self.start_train)
        self.select_model.setEnabled(True)
        self.check_gpu0.setCheckable(True)
        self.check_gpu1.setCheckable(True)

    def back_train(self):
        if self.process_train.state() != QProcess.NotRunning:
            self.process_train.close()
        self.process_gpu_info.close()
        self.process_gpu_utilization.close()
        self.parent.back_to_parent()
