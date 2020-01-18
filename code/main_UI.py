import sys
from pyqtgraph.Qt import QtGui
from online_pipeline.main_interface import MainInterface
from online_pipeline.config import stream_file
import logging

# Setup logging
root_logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler('main_ui.log')
file_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = MainInterface()
    myapp.show()
    app.exec_()
