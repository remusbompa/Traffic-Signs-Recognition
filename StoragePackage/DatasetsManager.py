from PyQt5.QtCore import QUrl, QFile, QTextStream, Qt
from PyQt5.QtGui import QIcon, QTextDocument, QPixmap
from PyQt5.QtWidgets import QComboBox, QTextEdit, QSizePolicy, QHBoxLayout, QLabel, QVBoxLayout, QCheckBox, \
    QStyledItemDelegate

dict_site = {"Belgium": "http://people.ee.ethz.ch/~timofter/traffic_signs/index.html",
                 "Belgium-Detection": "http://people.ee.ethz.ch/~timofter/traffic_signs/index.html",
                 "Croat": "http://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml",
                 "GTSDB": "http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset",
                 "GTSRB": "http://benchmark.ini.rub.de/?section=gtsrb&subsection=news",
                 "European": None,
                 "Swedish": "https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/"}


class DataSetsManager:

    instance = None

    def __init__(self):
        if DataSetsManager.instance:
            raise Exception("This class is a singleton!")
        else:
            DataSetsManager.instance = self

        self.dict_ds = {}
        for name in dict_site:
            ds_manager = DataSetManager(name)
            self.dict_ds[name] = ds_manager

    @staticmethod
    def get_instance():
        if DataSetsManager.instance:
            return DataSetsManager.instance
        return DataSetsManager()

    @staticmethod
    def get_data_set(name):
        return DataSetsManager.get_instance().dict_ds[name]

    @staticmethod
    def get_data_set_combo():
        select_data_set_list = QComboBox()
        item_delegate = QStyledItemDelegate()
        select_data_set_list.setItemDelegate(item_delegate)
        for name in dict_site:
            select_data_set_list.addItem(DataSetsManager.get_data_set(name).icon, name)
        return select_data_set_list

    @staticmethod
    def get_data_set_to_compare(hor_layout: QHBoxLayout):
        dict_check_boxes = {}
        for i, name in enumerate(dict_site):
            hor_icon_name = QHBoxLayout()
            hor_icon_name.addWidget(QLabel(name), 1)
            pix_map = QPixmap(f"../icons/{name}.png").scaled(60, 60, Qt.KeepAspectRatio)
            label_flag = QLabel()
            label_flag.setPixmap(pix_map)
            hor_icon_name.addWidget(label_flag, 2)
            vert_check_name_icon = QVBoxLayout()

            check_box = QCheckBox()
            check_box.setObjectName("StatisticsCheckbox")
            vert_check_name_icon.addLayout(hor_icon_name, 2)
            vert_check_name_icon.addWidget(check_box, 1, alignment=Qt.AlignCenter)

            hor_layout.addLayout(vert_check_name_icon, 1)
            dict_check_boxes[check_box] = name

            if i in [0, 2, 5]:
                check_box.setChecked(True)

            if i < len(dict_site):
                hor_layout.addStretch(1)

        return dict_check_boxes


class DataSetManager:
    def __init__(self, name):
        self.name = name
        self.weights = f"../weights/{name}.weights"
        self.cfg = f"../cfg/{name}.cfg"
        self.names = f"../data/{name}.names"
        self.html = f"../html_files/{name}.html"
        self.icon = QIcon(f"../icons/{name}.png")
        self.dist_site = dict_site[name]

        f = QFile(self.html)
        f.open(QFile.ReadOnly | QFile.Text)
        in_stream = QTextStream(f)
        self.html_text = in_stream.readAll()
        f.close()
        self.parameters = [f"./Tests/{name}/{name}.data",
                           f"./Tests/{name}/{name}-yolov3.cfg",
                           f"./Tests/{name}/backup/{name}-yolov3_last.weights"]
