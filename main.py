import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui_window import KineFaceWindow  # Cambiaremos el nombre de la clase en ui_window.py después
import ctypes

# Configuración para pantallas de alta resolución (DPI Scaling) en Windows
try:
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


def main():
    # ID único para que el sistema operativo identifique a KineFace por separado
    myappid = 'ucv.ciencias.kineface.v1'
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)

    # Seteamos el nombre de la aplicación
    app.setApplicationName("KineFace")

    # Cargamos el logo (asegúrate de tenerlo en assets/logo.png o usa el anterior por ahora)
    app.setWindowIcon(QIcon("assets/logo.png"))

    # Intentamos forzar el uso de OpenCL para acelerar OpenCV si hay GPU disponible
    try:
        import os
        os.environ["OPENCV_OPENCL_DEVICE"] = "gpu"
    except Exception:
        pass

    # Inicializamos la ventana con el nuevo nombre de clase
    window = KineFaceWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()