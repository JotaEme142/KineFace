import sys
import numpy as np
import cv2
import qtawesome as qta
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QProgressBar, QPushButton, QStackedWidget, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, pyqtSlot, QTimer

import pyqtgraph as pg
from core.camera import VideoThread
from core.face_analyzer import KineFaceAnalyzer


class KineFaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KineFace AI - Sistema de Monitoreo de Esfuerzo")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet("background-color: #0F111A; color: #E2E8F0;")

        # Instancia del analizador
        self.analyzer = KineFaceAnalyzer()
        self.effort_data = np.zeros(100)

        # Definición de Paleta Neon Bio-Tech
        self.color_ai = "#BB86FC"
        self.color_health = "#00E5FF"
        self.color_ok = "#4CAF50"
        self.color_off = "#374151"

        # Contenedor Principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 1. Configurar Header
        self.setup_header()

        # 2. Configurar Stack de Navegación
        self.stack = QStackedWidget()
        self.setup_calibration_page()
        self.setup_monitoring_page()
        self.main_layout.addWidget(self.stack)

        # 3. Hilo de Cámara
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # 4. Timer para la fase de Calibración
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self.performing_calibration)
        self.calib_counter = 0
        self.last_pts = None

    def setup_header(self):
        header = QHBoxLayout()
        title = QLabel("KINEFACE")
        title.setFont(QFont("Poppins", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.color_ai}; margin-left: 10px;")

        self.btn_mode = QPushButton(" MODO MANUAL")
        self.btn_mode.setIcon(qta.icon('fa5s.microchip', color=self.color_ai))
        self.btn_mode.setCheckable(True)
        self.btn_mode.setFixedSize(200, 40)
        self.btn_mode.setStyleSheet(f"""
            QPushButton {{
                background-color: #1E2130; border: 2px solid {self.color_ai};
                border-radius: 20px; font-weight: bold; color: {self.color_ai};
            }}
            QPushButton:checked {{
                background-color: {self.color_ai}; color: #0F111A;
            }}
        """)
        self.btn_mode.toggled.connect(self.on_mode_toggled)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.btn_mode)
        self.main_layout.addLayout(header)

    def setup_calibration_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        # Label de video (Tamaño controlado para evitar crecimiento errático)
        self.view_calib = QLabel("Esperando cámara...")
        self.view_calib.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_calib.setMinimumSize(720, 480)
        self.view_calib.setStyleSheet("border: 2px solid #1E2130; border-radius: 15px; background: #050505;")

        # Sidebar de validaciones
        side_frame = QFrame()
        side_frame.setFixedWidth(320)
        side_frame.setStyleSheet("background-color: #161926; border-radius: 15px;")
        side_layout = QVBoxLayout(side_frame)

        self.label_instr = QLabel("PASO 1: CALIBRACIÓN")
        self.label_instr.setStyleSheet(f"color: {self.color_health}; font-weight: bold; font-size: 18px;")
        self.label_instr.setAlignment(Qt.AlignmentFlag.AlignCenter)

        side_layout.addWidget(self.label_instr)
        side_layout.addSpacing(20)

        # Diccionario de Checks con QtAwesome
        self.checks = {
            "frontal": self.create_check_qta("Pose Frontal", side_layout),
            "eyes": self.create_check_qta("Ojos Abiertos", side_layout),
            "mouth": self.create_check_qta("Boca Cerrada", side_layout),
            "neutral": self.create_check_qta("Rostro Relajado", side_layout)
        }

        side_layout.addStretch()
        layout.addWidget(self.view_calib, 5)
        layout.addWidget(side_frame, 2)
        self.stack.addWidget(page)

    def create_check_qta(self, text, layout):
        container = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.circle', color=self.color_off).pixmap(24, 24))

        text_label = QLabel(text)
        text_label.setStyleSheet(f"color: {self.color_off}; font-size: 14px;")

        container.addWidget(icon_label)
        container.addWidget(text_label)
        container.addStretch()
        layout.addLayout(container)
        return {"icon": icon_label, "label": text_label}

    def setup_monitoring_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        left_col = QVBoxLayout()

        # Video de monitoreo
        self.view_monit = QLabel()
        self.view_monit.setFixedSize(400, 300)
        self.view_monit.setStyleSheet(f"border: 2px solid {self.color_ai}; border-radius: 10px; background: #000;")

        # Widget de Gráfica
        graph_container = QFrame()
        graph_container.setStyleSheet("background: #161926; border-radius: 15px; padding: 10px;")
        graph_layout = QVBoxLayout(graph_container)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('#161926')
        self.graph_widget.setYRange(0, 10)
        self.graph_widget.showGrid(x=False, y=True, alpha=0.3)
        self.curve = self.graph_widget.plot(pen=pg.mkPen(color=self.color_health, width=3))

        lbl_graph = QLabel("HISTORIAL DE ESFUERZO (ESCALA DE BORG)")
        lbl_graph.setStyleSheet("font-size: 10px; color: #888; letter-spacing: 1px;")
        graph_layout.addWidget(lbl_graph)
        graph_layout.addWidget(self.graph_widget)

        left_col.addWidget(self.view_monit)
        left_col.addWidget(graph_container)

        # Termómetro Vertical
        right_col = QVBoxLayout()
        self.effort_bar = QProgressBar()
        self.effort_bar.setOrientation(Qt.Orientation.Vertical)
        self.effort_bar.setFixedWidth(50)
        self.effort_bar.setTextVisible(False)
        self.effort_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #1E2130; border: 1px solid {self.color_off};
                border-radius: 25px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:1, x2:0, y2:0, 
                            stop:0 {self.color_health}, stop:0.7 {self.color_ai}, stop:1 #FF007A);
                border-radius: 23px;
            }}
        """)

        lbl_val = QLabel("NIVEL")
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_val.setStyleSheet(f"color: {self.color_ai}; font-weight: bold;")

        right_col.addWidget(lbl_val)
        right_col.addWidget(self.effort_bar)

        layout.addLayout(left_col, 4)
        layout.addLayout(right_col, 1)
        self.stack.addWidget(page)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        if not self.isVisible(): return

        res = self.analyzer.process_frame(cv_img)

        if res:
            pts, p, y = res
            self.last_pts = pts

            # Lógica de estados: Calibración vs Monitoreo
            if not self.analyzer.is_calibrated:
                is_ready, checks = self.analyzer.validate_neutral_state(pts, p, y)
                self.update_validation_visuals(checks)

                if is_ready:
                    if not self.calib_timer.isActive():
                        self.calib_timer.start(100)
                else:
                    self.calib_timer.stop()
                    self.calib_counter = 0
            else:
                effort = self.analyzer.compute_manual_effort(pts)
                self.update_monitoring_visuals(effort)

        # Renderizado de video
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

        if self.stack.currentIndex() == 0:
            target = self.view_calib.size()
            self.view_calib.setPixmap(QPixmap.fromImage(qt_img).scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                                                       Qt.TransformationMode.SmoothTransformation))
        else:
            target = self.view_monit.size()
            self.view_monit.setPixmap(QPixmap.fromImage(qt_img).scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                                                       Qt.TransformationMode.SmoothTransformation))

    def update_validation_visuals(self, checks):
        mapping = {"frontal": "frontal", "eyes_open": "eyes", "mouth_closed": "mouth", "neutral": "neutral"}
        for key, val in checks.items():
            if key in mapping:
                item = self.checks[mapping[key]]
                if val:
                    item["icon"].setPixmap(qta.icon('fa5s.check-circle', color=self.color_ok).pixmap(24, 24))
                    item["label"].setStyleSheet(f"color: {self.color_ok}; font-weight: bold;")
                else:
                    item["icon"].setPixmap(qta.icon('fa5s.circle', color=self.color_off).pixmap(24, 24))
                    item["label"].setStyleSheet(f"color: {self.color_off}; font-weight: normal;")

    def performing_calibration(self):
        """Cuenta regresiva de 3 segundos para calibrar"""
        self.calib_counter += 1
        self.label_instr.setText(f"CALIBRANDO...\n{max(0, 3 - self.calib_counter // 10)}s")

        if self.calib_counter >= 30:
            self.calib_timer.stop()
            if self.last_pts is not None:
                self.analyzer.calibrate(self.last_pts)
                self.stack.setCurrentIndex(1)  # Cambio a monitoreo

    def update_monitoring_visuals(self, effort):
        # Actualizar termómetro (0-10 -> 0-100)
        self.effort_bar.setValue(int(effort * 10))

        # Actualizar Gráfica con desplazamiento
        self.effort_data[:-1] = self.effort_data[1:]
        self.effort_data[-1] = effort
        self.curve.setData(self.effort_data)

    def on_mode_toggled(self, checked):
        """Maneja el switch entre Manual y Automático"""
        mode = " MODO AUTOMÁTICO" if checked else " MODO MANUAL"
        icon = 'fa5s.brain' if checked else 'fa5s.microchip'
        self.btn_mode.setText(mode)
        self.btn_mode.setIcon(qta.icon(icon, color="#0F111A" if checked else self.color_ai))

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()