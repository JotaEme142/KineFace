import cv2
import numpy as np
import os
from PyQt6.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    # Señal que transporta el frame crudo de la cámara
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self._run_flag = True
        self.camera_index = camera_index

    def run(self):
        source = self.camera_index
        # Mantenemos la lógica del archivo externo para facilitar cambios de cámara
        base_dir = os.path.abspath(".")
        file_path = os.path.join(base_dir, "camara.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read().strip()
                if content:
                    source = int(content) if content.isdigit() else content

        cap = cv2.VideoCapture(source)
        # Seteamos resolución HD para mejor detección de micro-expresiones
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            else:
                break

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()