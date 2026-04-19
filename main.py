import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui_window import MainWindow
import ctypes

# Configuración para pantallas de alta resolución (DPI Scaling)
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

def main():
    myappid = 'kineface'
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("assets/logo.png"))

    try:
        import os
        os.environ["OPENCV_OPENCL_DEVICE"] = "gpu"
    except:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()