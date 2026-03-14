import os

from .simulator import Simulator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_FILES_DIR = os.path.join(CURRENT_DIR, "run_files")
RUN_EXT = "run"
