from PyQt5 import QtWidgets

from controller import MainWindow_controller
from dogcat import Net

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())
