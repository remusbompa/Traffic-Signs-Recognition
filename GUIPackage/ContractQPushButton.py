from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QComboBox


class ContractQPushButton(QHBoxLayout):
    def __init__(self, button: QWidget, parent=None):
        super(ContractQPushButton, self).__init__(parent)
        self.addStretch(1)
        self.addWidget(button, stretch=2)
        self.addStretch(1)


class ContractQText(QHBoxLayout):
    def __init__(self, text_edit: QWidget, parent=None):
        super(ContractQText, self).__init__(parent)
        self.addStretch(1)
        self.addWidget(text_edit, stretch=2)
        self.addStretch(1)


class SelectQPushButton(QHBoxLayout):
    def __init__(self, button: QWidget, parent=None):
        super(SelectQPushButton, self).__init__(parent)
        self.addWidget(button, stretch=2)
        self.addStretch(2)


class SelectQText(QHBoxLayout):
    def __init__(self, text_edit: QWidget, parent=None):
        super(SelectQText, self).__init__(parent)
        self.addWidget(text_edit, stretch=2)
        self.addStretch(2)


class SelectQCombo(QHBoxLayout):
    def __init__(self, text_edit: QLabel, combo: QComboBox, parent=None):
        super(SelectQCombo, self).__init__(parent)
        text_edit.setObjectName("SelectLabel")
        self.addWidget(text_edit, stretch=1)
        self.addWidget(combo, stretch=3)
        self.addStretch(4)
