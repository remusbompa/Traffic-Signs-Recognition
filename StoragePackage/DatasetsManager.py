from PyQt5.QtGui import QPixmap

dict_site = {"Belgium": "http://people.ee.ethz.ch/~timofter/traffic_signs/index.html",
                 "Belgium-Detection": "http://people.ee.ethz.ch/~timofter/traffic_signs/index.html",
                 "Croat-Detection": "http://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml",
                 "GTSDB": "http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset",
                 "GTSRB": "http://benchmark.ini.rub.de/?section=gtsrb&subsection=news",
                 "European": None,
                 "Swedish:" : "https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/"}


class DataSetsManager:

    def __init__(self):
        self.list_ds = []
        for name in dict_site:
            self.list_ds.append(DataSetManager(name))


class DataSetManager:
    def __init__(self, name):
        self.name = name
        self.weights = f"../weights/{name}.weights"
        self.cfg = f"../cfg/{name}.cfg"
        self.names = f"../data/{name}.names"
        self.web_site = dict_site[name]
        self.icon = QPixmap(f"../icons/{name}.png")
