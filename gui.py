import sys

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QLabel, QCheckBox, QRadioButton
from ssa import calculate_similarity, draw_graph
from sim_max_b import sim_max_b, draw_graph_sim

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Semantic similarity'
        self.left = 300
        self.top = 300
        self.width = 425
        self.height = 250
        self.textbox1 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)
        self.l1 = QLabel(self)
        self.l2 = QLabel(self)
        self.l3 = QLabel(self)
        self.l4 = QLabel(self)
        self.checkbox = QCheckBox(self)
        self.button = QPushButton('Calculate', self)
        self.b1_ssa = QRadioButton(self)
        self.b2_maxb = QRadioButton(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.b1_ssa.move(20, 5)
        self.b1_ssa.setText("SSA")
        self.b1_ssa.setChecked(True)
        self.b2_maxb.move(20, 25)
        self.b2_maxb.setText("Sim Max B")

        self.l1.move(20, 53)
        self.l1.setText("word 1: ")
        self.textbox1.move(120, 60)
        self.textbox1.resize(280, 20)

        self.l2.move(20, 83)
        self.l2.setText("word 2: ")
        self.textbox2.move(120, 90)
        self.textbox2.resize(280, 20)

        self.l3.move(20, 113)
        self.l3.setText("draw graph")

        self.checkbox.move(120, 113)
        self.button.move(20, 163)

        self.l4.move(20, 203)
        self.l4.setText("result:")
        self.textbox3.move(120, 210)
        self.textbox3.resize(280, 20)
        self.textbox3.setReadOnly(True)

        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        word1 = self.textbox1.text()
        word2 = self.textbox2.text()
        if self.b1_ssa.isChecked():
            g, dist, path = calculate_similarity(word1, word2)
            self.textbox3.setText(str(np.round(dist * 10, 2)))
            if self.checkbox.isChecked():
                draw_graph(g, word1, word2, path)
        elif self.b2_maxb.isChecked():
            _, _, sim, nx_graph, _, path = sim_max_b(word1, word2)
            self.textbox3.setText(str(np.round(sim * 10, 2)))
            if self.checkbox.isChecked():
                draw_graph_sim(nx_graph, path, word1, word2, sim)


def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
