import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_FILES_DIR = os.path.join(CURRENT_DIR, "map_files")
EXT = "npz"  # File extension for map files
TEMPLATES_FILE_NAME = "template_{}." + EXT
