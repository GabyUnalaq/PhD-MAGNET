import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from src.map.builder import MapBuilderWindow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore-validity", action="store_true", help="Allow saving maps that fail validity checks")
    args = parser.parse_args()

    try:
        app = QApplication(sys.argv)
        window = MapBuilderWindow(ignore_validity=args.ignore_validity)
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)
