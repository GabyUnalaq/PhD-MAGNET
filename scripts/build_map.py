import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from src.map.builder import MapBuilderWindow


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MapBuilderWindow()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)
