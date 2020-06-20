from PyQt5.QtCore import QCoreApplication, Qt, QProcess, QUrl, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont, QStandardItemModel, QStandardItem, QTextDocument, QPalette, QTextCursor, \
    QPainter, QColor
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel, QComboBox, \
    QTextEdit, QWidget, QCheckBox, QSizePolicy, qApp

from ServerPackage.ResultsManager import ResultsManager
from StoragePackage.DatasetsManager import DataSetsManager
from PyQt5.QtChart import QChart, QChartView, QHorizontalBarSeries, QBarSet, QBarCategoryAxis, QValueAxis, QBarSeries, \
    QPieSeries, QPieSlice
import pyqtgraph as pg


def resize_window(main_win: QMainWindow, percent: float = 0.9):
    available_size = qApp.desktop().availableGeometry().size()
    width = available_size.width()
    height = available_size.height()
    width *= percent
    height *= percent
    new_size = QSize(width, height)
    main_win.resize(new_size)


class StatisticsGUI(QMainWindow):
    def __init__(self, parent=None):
        super(StatisticsGUI, self).__init__(parent)
        self.title = 'Statistics Traffic IA'
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
    def __init__(self, parent: StatisticsGUI = None):
        super().__init__()
        self.parent = parent
        self.setWindowIcon(QIcon('../images/Logo.png'))
        layout = QVBoxLayout()
        # first line: select type of comparison, inter-dataset or intra-dataset
        self.layout_horizontal1 = QHBoxLayout()
        self.comp_type_label = QLabel("Select comparison type: ")
        self.comp_type = QComboBox()
        self.comp_type.addItem("Inter-Dataset Comparison")
        self.comp_type.addItem("Intra-Dataset Comparison")
        self.comp_type_crt_text = self.comp_type.currentText()
        self.comp_type.currentTextChanged.connect(self.comp_type_changed)
        self.layout_horizontal1.addWidget(self.comp_type_label, 1)
        self.layout_horizontal1.addWidget(self.comp_type, 2, alignment=Qt.AlignLeft)
        self.layout_horizontal1.addStretch(3)

        layout.addLayout(self.layout_horizontal1, 2)
        layout.addStretch(1)
        # second line: check datasets to compare (inter) / dataset to compare classes (intra)
        self.layout_vertical2_inter = QVBoxLayout()
        self.check_ds = QLabel('Check datasets to compare: ')
        self.layout_horizontal2 = QHBoxLayout()
        self.check_boxes = DataSetsManager.get_data_set_to_compare(self.layout_horizontal2)
        self.layout_vertical2_inter.addWidget(self.check_ds, 1, alignment=Qt.AlignLeft)
        self.layout_vertical2_inter.addLayout(self.layout_horizontal2, 2)

        self.wid_1_inter = QWidget()
        self.wid_1_inter.setLayout(self.layout_vertical2_inter)
        layout.addWidget(self.wid_1_inter, 2)
        layout.addStretch(1)

        # for intra
        self.layout_vertical2_intra = QHBoxLayout()
        self.select_ds = QLabel("Select dataset: ")
        self.list_ds = DataSetsManager.get_data_set_combo()
        self.list_ds.setCurrentIndex(0)
        self.intra_ds_current = self.list_ds.itemText(0)
        self.list_ds.currentTextChanged.connect(self.dataset_changed)
        self.layout_vertical2_intra.addWidget(self.select_ds, 1)
        self.layout_vertical2_intra.addWidget(self.list_ds, 2, alignment=Qt.AlignLeft)
        self.layout_vertical2_intra.addStretch(3)
        self.wid_1_intra = QWidget()
        self.wid_1_intra.setLayout(self.layout_vertical2_intra)
        # third line: select metric for comparison
        self.layout_hor3_inter = QHBoxLayout()
        self.select_metric_label = QLabel("Select metric: ")
        self.select_metric_inter = QComboBox()
        self.dict_metric_inter = {"Number of detections": ("detections_count", 'd'),
                             "Number of detections (unique per box)": ("unique_truth_count", 'd'),
                             "Precision": ("precision", 'f'),
                             "Recall": ("recall", 'f'),
                             "F1 score": ("F1-score", 'f'),
                             "True Positives": ("TP", 'd'),
                             "False Positives": ("FP", 'd'),
                             "False Negatives": ("FN", 'd'),
                             "Average IoU": ("average_IoU", 'p'),
                             "mean Average Precision": ("mAP", 'p'),
                             "Detection time": ("Time_seconds", 'd')
                             }
        for metric in self.dict_metric_inter:
            self.select_metric_inter.addItem(metric)
        self.layout_hor3_inter.addWidget(self.select_metric_label, 1)
        self.layout_hor3_inter.addWidget(self.select_metric_inter, 2, alignment=Qt.AlignLeft)
        self.layout_hor3_inter.addStretch(3)
        self.wid_2_inter = QWidget()
        self.wid_2_inter.setLayout(self.layout_hor3_inter)
        layout.addWidget(self.wid_2_inter, 2)

        # for intra
        self.layout_hor3_intra = QHBoxLayout()
        self.select_metric_intra = QComboBox()
        self.dict_metric_intra = {"Average precision": ("ap", 'p'),
                             "Number of detections (unique per box)": ("TC", 'd'),
                             "Precision": ("Precision", 'f'),
                             "Recall": ("Recall", 'f'),
                             "F1 score": ("F1-score", 'f'),
                             "True Positives": ("TP", 'd'),
                             "False Positives": ("FP", 'd'),
                             "False Negatives": ("FN", 'd'),
                             "Average IoU": ("Avg_IOU", 'p')
                             }
        for metric in self.dict_metric_intra:
            self.select_metric_intra.addItem(metric)
        self.select_metric_label_intra = QLabel("Select metric: ")
        self.layout_hor3_intra.addWidget(self.select_metric_label_intra, 1)
        self.layout_hor3_intra.addWidget(self.select_metric_intra, 2, alignment=Qt.AlignLeft)
        self.layout_hor3_intra.addStretch(3)
        self.wid_2_intra = QWidget()
        self.wid_2_intra.setLayout(self.layout_hor3_intra)

        # fourth line: select chart type
        self.layout_hor4 = QHBoxLayout()
        self.select_chart = QComboBox()
        self.select_chart.addItem("Horizontal Bar chart")
        self.select_chart.addItem("Vertical Bar chart")
        self.select_chart.addItem("Pie chart")
        self.select_chart_label = QLabel("Select chart type: ")
        self.layout_hor4.addWidget(self.select_chart_label, 1)
        self.layout_hor4.addWidget(self.select_chart, 2, alignment=Qt.AlignLeft)
        self.layout_hor4.addStretch(3)
        layout.addLayout(self.layout_hor4, 1)
        # fifth line: chart
        self.graph_layout = QHBoxLayout()
        self.crt_chart = QWidget()
        self.graph_layout.addWidget(self.crt_chart)
        layout.addLayout(self.graph_layout, 10)
        # sixth line: Run and Back
        self.back_button = QPushButton("Back")
        self.run_button = QPushButton("Run")
        self.hor_box = QHBoxLayout()
        self.hor_box.addStretch(9)
        self.hor_box.addWidget(self.back_button, 1)
        self.hor_box.addWidget(self.run_button, 1)
        self.back_button.clicked.connect(self.back_statistics)
        self.run_button.clicked.connect(self.run_statistics)
        layout.addLayout(self.hor_box, 1)

        self.wid_1_inter.setObjectName("StatisticsLayout")
        self.wid_1_intra.setObjectName("StatisticsLayout")
        self.wid_2_inter.setObjectName("StatisticsLayout")
        self.wid_2_intra.setObjectName("StatisticsLayout")

        self.run_statistics()
        self.setLayout(layout)

    def comp_type_changed(self, text):
        if text != self.comp_type_crt_text:
            self.comp_type_crt_text = text
            if text[:5] == "Inter":
                self.wid_1_intra.hide()
                self.wid_2_intra.hide()

                self.wid_1_inter.show()
                self.wid_2_inter.show()

                self.layout().replaceWidget(self.wid_1_intra, self.wid_1_inter)
                self.layout().replaceWidget(self.wid_2_intra, self.wid_2_inter)

            elif text[:5] == "Intra":
                self.wid_1_inter.hide()
                self.wid_2_inter.hide()

                self.wid_1_intra.show()
                self.wid_2_intra.show()

                self.layout().replaceWidget(self.wid_1_inter, self.wid_1_intra)
                self.layout().replaceWidget(self.wid_2_inter, self.wid_2_intra)

    def dataset_changed(self, text):
        self.intra_ds_current = text

    def back_statistics(self):
        self.parent.back_to_parent()

    def run_statistics(self):
        text = self.comp_type_crt_text
        x_vals = []
        y_vals = []
        if text[:5] == "Inter":
            data_sets = []
            for check_box in self.check_boxes:
                if check_box.isChecked():
                    data_sets.append(self.check_boxes[check_box])

            metric = self.select_metric_inter.currentText()
            metric_alias = self.dict_metric_inter[metric]
            res = ResultsManager.get_results_inter(metric_alias, data_sets)
            # self.graphWidget.plot(data_sets, res)
            x_vals = data_sets
            y_vals = res
        elif text[:5] == "Intra":
            ds_name = self.intra_ds_current
            metric = self.select_metric_intra.currentText()
            metric_alias = self.dict_metric_intra[metric]
            names, res = ResultsManager.get_results_intra(metric_alias, ds_name)
            # self.graphWidget.plot(names, res)
            x_vals = names
            y_vals = res
        #horizontal bar
        if self.select_chart.currentIndex() == 0:
            chart = QChart(flags=Qt.WindowFlags())
            series = QHorizontalBarSeries()
            for i in range(len(x_vals)):
                name = x_vals[i]
                set0 = QBarSet(name)
                set0.append(y_vals[i])
                series.append(set0)
            chart.addSeries(series)
            chart.setTitle(f"Comparison by {self.select_metric_inter.currentText()}")
            chart.setAnimationOptions(QChart.SeriesAnimations)

            chart_view = QChartView(chart)
            self.graph_layout.replaceWidget(self.crt_chart, chart_view)
            self.crt_chart = chart_view

            axisX = QValueAxis()
            chart.addAxis(axisX, Qt.AlignBottom)
            series.attachAxis(axisX)

            axisX.applyNiceNumbers()

            chart.legend().setVisible(True)
            chart.legend().setAlignment(Qt.AlignBottom)

            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setBackgroundBrush(QColor(0, 0, 0, 255))
            chart.setBackgroundBrush(QColor(255, 255, 0, 255))
        # vertical bar
        elif self.select_chart.currentIndex() == 1:
            chart = QChart(flags=Qt.WindowFlags())
            series = QBarSeries()
            for i in range(len(x_vals)):
                name = x_vals[i]
                set0 = QBarSet(name)
                set0.append(y_vals[i])
                series.append(set0)
            chart.addSeries(series)
            chart.setTitle(f"Comparison by {self.select_metric_inter.currentText()}")
            chart.setAnimationOptions(QChart.SeriesAnimations)

            chart_view = QChartView(chart)
            self.graph_layout.replaceWidget(self.crt_chart, chart_view)
            self.crt_chart = chart_view

            axisY = QValueAxis()
            chart.addAxis(axisY, Qt.AlignLeft)
            series.attachAxis(axisY)
            axisY.applyNiceNumbers()

            chart.legend().setVisible(True)
            chart.legend().setAlignment(Qt.AlignBottom)

            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setBackgroundBrush(QColor(0, 0, 0, 255))
            chart.setBackgroundBrush(QColor(255, 255, 0, 255))
        # pie chart
        elif self.select_chart.currentIndex() == 2:
            chart = QChart(flags=Qt.WindowFlags())
            series = QPieSeries()
            for i in range(len(x_vals)):
                series.append(x_vals[i], y_vals[i])
            chart.addSeries(series)
            chart.setTitle(f"Comparison by {self.select_metric_inter.currentText()}")
            chart.setAnimationOptions(QChart.SeriesAnimations)
            chart.legend().setAlignment(Qt.AlignBottom)
            # chart.legend().setFont(QFont("Arial", 12))

            chart_view = QChartView(chart)
            self.graph_layout.replaceWidget(self.crt_chart, chart_view)
            self.crt_chart = chart_view

            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setBackgroundBrush(QColor(0, 0, 0, 255))
            chart.setBackgroundBrush(QColor(255, 255, 0, 255))
