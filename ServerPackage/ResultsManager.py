import xml.etree.ElementTree as ET

"""
dict_data_sets = ["Belgium", "Belgium-Detection", "Croat",
             "GTSDB", "GTSRB", "European", "Swedish"]
"""

dict_data_sets = ["Belgium", "Croat", "European", "Swedish"]


def parse_xml(my_file):
    tree = ET.parse(my_file)
    root = tree.getroot()
    my_dict = {}
    for child in root:
        if child.tag == "Classes":
            classes_dict = {}
            for cls in child:
                cls_dict = {}
                cls_id = -1
                for atr in cls:
                    if atr.tag == "class_id":
                        cls_id = int(atr.text)
                    cls_dict[atr.tag] = atr.text
                classes_dict[cls_id] = cls_dict
            my_dict["Classes"] = classes_dict
        else:
            my_dict[child.tag] = child.text
    return my_dict


class ResultsManager:

    instance = None

    def __init__(self):
        if ResultsManager.instance:
            raise Exception("This class is a singleton!")
        else:
            ResultsManager.instance = self

        self.dict_ds = {}
        for name in dict_data_sets:
            xml_dict = parse_xml(f"../results/{name}.xml")
            self.dict_ds[name] = xml_dict

    @staticmethod
    def get_instance():
        if ResultsManager.instance:
            return ResultsManager.instance
        return ResultsManager()

    @staticmethod
    def get_data_set_xml(name):
        return ResultsManager.get_instance().dict_ds[name]

    @staticmethod
    def get_results_inter(metric, data_sets):
        res_list = []
        for name in data_sets:
            ds_xml = ResultsManager.get_data_set_xml(name)
            if metric[1] == 'd':
                res_list.append(int(ds_xml[metric[0]]))
            else:
                res_list.append(float(ds_xml[metric[0]].strip('%')))
        return res_list

    @staticmethod
    def get_results_intra(metric, name):
        ds_xml = ResultsManager.get_data_set_xml(name)
        classes = ds_xml["Classes"]
        names = []
        res_list = []
        for cls in classes:
            dict_cls = classes[cls]
            names.append(dict_cls["name"])
            if metric[1] == 'd':
                res_list.append(int(dict_cls[metric[0]]))
            else:
                res_list.append(float(dict_cls[metric[0]].strip('%')))
        return names, res_list
