import cv2
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QSlider, QGroupBox, QPushButton, QCheckBox, QGridLayout, QApplication,
                             QSizePolicy, QMessageBox, QToolBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
import qtawesome as qta
import numpy as np
import winsound

# Importaciones modulares
from core.camera import VideoThread
from core.face_analyzer import FaceAnalyzer

import json
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KineFace - Monitoreo de Rehabilitación")
        self.setWindowIcon(QIcon("assets/logo.png"))
        self.resize(1280, 720)
        self.setStyleSheet("""
            * {
                outline: none;
            }
            QMainWindow, QMessageBox { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            
            QMessageBox QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 4px;
                padding: 6px 15px;
                min-width: 70px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover { background-color: #0062a3; }

            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 20px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; 
                padding: 0 10px;
                color: white; 
                background-color: #1e1e1e; 
            }
        """)

        self.analyzer = FaceAnalyzer()
        self.thread = VideoThread(camera_index=0)

        self.current_processed_frame = None
        self.brightness_value = 0
        self.contrast_value = 0
        self.check_labels = {}
        self.last_valid_pts = None

        self.frame_counter = 0  # Necesario para el conteo de frames
        self.last_results = None  # Necesario para guardar la caché del análisis

        self.auto_capture_timer = QTimer()
        self.auto_capture_timer.setInterval(1000)  # 1 segundo (1000 ms)
        self.auto_capture_timer.timeout.connect(self.tick_countdown)

        self.countdown_value = 3  # Segundos para esperar
        self.is_auto_capturing = False  # Bandera para saber si estamos contando
        self.showing_preview = False  # Bandera para evitar bucle infinito
        self.preview_raw_img = None # Variable para la imagen estática en edición
        self.preview_mask = None

        self.debug_mode = False

        self.CONFIG_FILE = "thresholds.json"
        self.init_ui()
        self.load_config()

        self.thread.change_pixmap_signal.connect(self.update_image)
        # self.thread.start() <--- Descomentar para iniciar la camara al ejecutar
        self.camera_active = False

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setup_left_debug_panel()

        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000000;")
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("Esperando cámara...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #666; font-size: 16px;")
        self.image_label.setFixedSize(800, 600)
        video_layout.addWidget(self.image_label)

        self.main_layout.addWidget(self.video_container, stretch=3)

        self.sidebar_container = QWidget()
        self.sidebar_container.setObjectName("RightPanel")
        self.sidebar_container.setStyleSheet("#RightPanel { background-color: #1e1e1e; border-left: 1px solid #333; }")
        self.controls_layout = QVBoxLayout(self.sidebar_container)
        self.controls_layout.setContentsMargins(20, 20, 20, 20)
        self.controls_layout.setSpacing(15)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.main_layout.addWidget(self.sidebar_container, stretch=1)

        self.btn_camera_toggle = QPushButton("INICIAR CÁMARA")
        self.btn_camera_toggle.setFixedSize(180, 40)
        self.btn_camera_toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_camera_toggle.setStyleSheet("""
            QPushButton {
                background-color: #007acc; 
                color: white; 
                font-weight: bold; 
                font-size: 14px; 
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #0062a3; }
            QPushButton:pressed { background-color: #005a96; }
        """)
        self.btn_camera_toggle.clicked.connect(self.toggle_camera)
        self.controls_layout.addWidget(self.btn_camera_toggle, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.setup_image_controls()
        self.setup_checklist_controls()
        self.setup_advanced_controls()
        self.controls_layout.addStretch()

    def setup_left_debug_panel(self):
        self.left_debug_container = QWidget()
        self.left_debug_container.setFixedWidth(280)
        self.left_debug_container.setObjectName("LeftPanel")
        self.left_debug_container.setStyleSheet(
            "#LeftPanel { background-color: #252526; border-right: 1px solid #333; }")
        layout = QVBoxLayout(self.left_debug_container)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # TÍTULO Y ACORDEÓN DE CALIBRACIÓN
        title = QLabel("CALIBRACIÓN")
        title.setStyleSheet("font-weight: bold; color: #aaa; font-size: 12px;")
        layout.addWidget(title)

        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet("""
            QToolBox::tab { background-color: #333; color: #e0e0e0; border-radius: 4px; }
            QToolBox::tab:selected { font-weight: bold; background-color: #007acc; color: white; }
        """)

        self.threshold_sliders = {}

        slider_groups = {
            "Estimación de Pose": [
                ("yaw", "Giro (Yaw)", 5, 45, 15, 1),
                ("pitch", "Inclin. (Pitch)", 5, 45, 15, 1),
                ("roll", "Rotación (Roll)", 5, 45, 12, 1),
            ],
            "Expresión Facial": [
                ("mouth", "Boca (Ratio)", 5, 80, 30, 100),
                ("smile", "Sonrisa (Max)", 25, 60, 43, 100),
                ("frown", "Ceño (Min)", 50, 90, 80, 100),
                ("eyebrow", "Cejas (Max)", 8, 20, 16, 100),
            ],
            "Ojos y Mirada": [
                ("ear", "Ojos (EAR)", 10, 40, 20, 100),
                ("gaze", "Mirada (Gaze)", 1, 30, 8, 100),
            ],
            "Calidad y Entorno": [
                ("blur", "Nitidez (Blur)", 0, 100, 30, 10),
                ("occlusion", "Oclusión (Color)", 10, 100, 50, 1),
                ("bg", "Fondo (STD)", 10, 100, 40, 1),
            ],
            "Posicionamiento": [
                ("dist_min", "Dist. Min", 5, 30, 12, 100),
                ("dist_max", "Dist. Max", 20, 50, 22, 100),
            ]
        }

        for group_name, sliders in slider_groups.items():
            page_widget = QWidget()
            page_layout = QVBoxLayout(page_widget)
            page_layout.setContentsMargins(5, 10, 5, 10)

            page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            for key, name, v_min, v_max, v_def, scale in sliders:
                lbl_box = QHBoxLayout()
                lbl_name = QLabel(name)
                lbl_name.setStyleSheet("color: #aaa; font-size: 11px;")
                lbl_val = QLabel(str(v_def / scale if scale > 1 else v_def))
                lbl_val.setStyleSheet("color: #4dc4ff; font-weight: bold; font-size: 11px;")

                lbl_box.addWidget(lbl_name)
                lbl_box.addStretch()
                lbl_box.addWidget(lbl_val)
                page_layout.addLayout(lbl_box)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(v_min, v_max)
                slider.setValue(v_def)
                slider.valueChanged.connect(lambda v, k=key, s=scale, l=lbl_val: self.on_threshold_changed(k, v, s, l))

                page_layout.addWidget(slider)
                self.threshold_sliders[key] = (slider, scale, lbl_val)
                page_layout.addSpacing(5)

            page_layout.addStretch()

            self.toolbox.addItem(page_widget, group_name)

        layout.addWidget(self.toolbox)

        # SEPARADOR PARA AJUSTAR ESPACIO
        layout.addStretch()

        # DATOS EN TIEMPO REAL (Anclados abajo)
        self.debug_group = QGroupBox("Datos en Tiempo Real")
        self.debug_group.setStyleSheet("""
            QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; color: #4dc4ff; }
            QLabel { color: #00ff00; font-family: 'Consolas', monospace; font-size: 11px; }
        """)
        debug_layout = QGridLayout()
        debug_layout.setHorizontalSpacing(10)

        self.lbl_yaw = QLabel("Yaw: 0")
        self.lbl_pitch = QLabel("Pitch: 0")
        self.lbl_roll = QLabel("Roll: 0")
        self.lbl_blur = QLabel("Blur: 0")
        self.lbl_light = QLabel("Luz: 0")
        self.lbl_light_dev = QLabel("Dev Luz: 0")
        self.lbl_gaze = QLabel("Gaze: 0")
        self.lbl_mouth = QLabel("Boca: 0")
        self.lbl_smile = QLabel("Sonrisa: 0")

        # Ancho fijo para evitar temblores al cambiar los números
        for lbl in [self.lbl_yaw, self.lbl_pitch, self.lbl_roll, self.lbl_blur,
                    self.lbl_light, self.lbl_light_dev, self.lbl_gaze, self.lbl_mouth, self.lbl_smile]:
            lbl.setMinimumWidth(110)

        debug_layout.addWidget(self.lbl_yaw, 0, 0)
        debug_layout.addWidget(self.lbl_pitch, 0, 1)
        debug_layout.addWidget(self.lbl_roll, 1, 0)
        debug_layout.addWidget(self.lbl_blur, 1, 1)
        debug_layout.addWidget(self.lbl_light, 2, 0)
        debug_layout.addWidget(self.lbl_light_dev, 2, 1)
        debug_layout.addWidget(self.lbl_gaze, 3, 0)
        debug_layout.addWidget(self.lbl_mouth, 4, 0)
        debug_layout.addWidget(self.lbl_smile, 4, 1)

        self.debug_group.setLayout(debug_layout)
        layout.addWidget(self.debug_group)

        # Ocultamos el panel al iniciar
        self.left_debug_container.setVisible(False)
        self.main_layout.insertWidget(0, self.left_debug_container)

    def on_threshold_changed(self, key, value, scale, label):
        val = value / scale
        label.setText(f"{val:.2f}" if scale > 1 else str(value))
        self.analyzer.set_thresholds({key: val})
        self.save_config()

    def save_config(self):
        config = {}
        for key, (slider, scale, label) in self.threshold_sliders.items():
            config[key] = slider.value()
        
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error al guardar config: {e}")

    def load_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return

        try:
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)

            for key, config_item in self.threshold_sliders.items():
                if key in config:
                    slider, scale, label = config_item
                    slider.blockSignals(True)
                    slider.setValue(config[key])
                    slider.blockSignals(False)
                    
                    val = config[key] / scale
                    label.setText(f"{val:.2f}" if scale > 1 else str(config[key]))
                    self.analyzer.set_thresholds({key: val})
        except Exception as e:
            print(f"Error al cargar config: {e}")

    def setup_image_controls(self):
        group = QGroupBox("AJUSTES DE IMAGEN")
        layout = QGridLayout()
        layout.setVerticalSpacing(15)

        self.lbl_brightness_txt = QLabel("Brillo")
        self.lbl_brightness_txt.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        self.lbl_brightness_val = QLabel("0")
        self.lbl_brightness_val.setStyleSheet("color: #fff; font-weight: bold;")

        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.valueChanged.connect(self.update_brightness)

        layout.addWidget(self.lbl_brightness_txt, 0, 0)
        layout.addWidget(self.lbl_brightness_val, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.slider_brightness, 1, 0, 1, 2)

        self.lbl_contrast_txt = QLabel("Contraste")
        self.lbl_contrast_txt.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        self.lbl_contrast_val = QLabel("0")
        self.lbl_contrast_val.setStyleSheet("color: #fff; font-weight: bold;")

        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(-100, 100)
        self.slider_contrast.setValue(0)
        self.slider_contrast.valueChanged.connect(self.update_contrast)

        layout.addWidget(self.lbl_contrast_txt, 2, 0)
        layout.addWidget(self.lbl_contrast_val, 2, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.slider_contrast, 3, 0, 1, 2)

        self.btn_reset_img = QPushButton("Restaurar")
        self.btn_reset_img.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_reset_img.setStyleSheet("""
            QPushButton {
                background-color: transparent; 
                border: 1px solid #555; 
                color: #aaa; 
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
            }
            QPushButton:hover { border-color: #888; color: #fff; }
        """)
        self.btn_reset_img.setFixedWidth(100)
        self.btn_reset_img.clicked.connect(self.reset_image_controls)
        layout.addWidget(self.btn_reset_img, 4, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        group.setLayout(layout)
        self.controls_layout.addWidget(group)

    def update_brightness(self, value):
        self.brightness_value = value
        texto = f"+{value}" if value > 0 else f"{value}"
        self.lbl_brightness_val.setText(texto)
        if self.showing_preview: self.update_preview_display()

    def update_contrast(self, value):
        self.contrast_value = value
        texto = f"+{value}" if value > 0 else f"{value}"
        self.lbl_contrast_val.setText(texto)
        if self.showing_preview: self.update_preview_display()

    def reset_image_controls(self):
        self.slider_brightness.setValue(0)
        self.slider_contrast.setValue(0)

    def setup_checklist_controls(self):
        self.group_checklist = QGroupBox("VALIDACIÓN DE CALIDAD")
        layout = QVBoxLayout()
        layout.setSpacing(5)

        self.checks_config = {
            "face_detected": ("mdi.face-recognition", "Rostro Detectado"),
            "no_hands": ("mdi.hand-right", "Rostro Sin Manos"),
            "no_occlusion": ("mdi.glasses", "Rostro Visible (Sin Objetos)"),
            "centered": ("mdi.crosshairs", "Rostro Centrado"),
            "distance": ("mdi.human-male-height", "Distancia Correcta"),
            "pose": ("mdi.axis-arrow", "Postura Correcta"),
            "gaze": ("mdi.eye-check", "Mirada al Frente"),
            "eyes": ("mdi.eye-outline", "Ojos Abiertos"),
            "mouth": ("fa5.meh-blank", "Boca Cerrada"),
            "neutral_exp": ("mdi.emoticon-neutral-outline", "Expresión Neutra"),
            "lighting": ("mdi.lightbulb-on-outline", "Iluminación General"),
            "lighting_uniform": ("mdi.theme-light-dark", "Luz Uniforme"),
            "sharpness": ("mdi.image-filter-center-focus", "Enfoque"),
            "background_ok": ("mdi.texture-box", "Fondo Uniforme")
        }

        self.check_widgets = {}

        for key, (icon_name, text) in self.checks_config.items():
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)
            row_layout.setContentsMargins(5, 0, 5, 0)

            # Contenedor del Icono
            lbl_icon = QLabel()
            lbl_icon.setFixedSize(20, 20)

            # Contenedor del Texto
            lbl_text = QLabel(text)
            lbl_text.setStyleSheet("color: #666; font-size: 13px;")

            row_layout.addWidget(lbl_icon)
            row_layout.addWidget(lbl_text)
            row_layout.addStretch()

            layout.addLayout(row_layout)

            # Guardamos las referencias para actualizarlas en tiempo real
            self.check_widgets[key] = (lbl_icon, lbl_text, icon_name)

        layout.addSpacing(10)
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        self.btn_capture = QPushButton("CAPTURAR ID")
        self.btn_capture.setMinimumHeight(40)
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.handle_capture_click)
        self.btn_capture.setStyleSheet("""
            QPushButton { background-color: #333; color: #555; border-radius: 6px; font-weight: bold; border: none;}
            QPushButton:enabled { background-color: #28a745; color: white; }
            QPushButton:enabled:hover { background-color: #218838; }
        """)
        self.buttons_layout.addWidget(self.btn_capture)

        self.btn_save_final = QPushButton("GUARDAR")
        self.btn_save_final.setMinimumHeight(40)
        self.btn_save_final.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_final.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_save_final.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; border-radius: 6px; font-weight: bold; border: none; }
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_save_final.clicked.connect(self.final_save_action)
        self.btn_save_final.setVisible(False)
        self.buttons_layout.addWidget(self.btn_save_final)

        self.btn_capture.setIcon(qta.icon('fa5s.camera', color='#555'))
        self.btn_save_final.setIcon(qta.icon('fa5s.save', color='white'))

        self.group_checklist.setLayout(layout)
        self.controls_layout.addWidget(self.group_checklist)

        self.controls_layout.addSpacing(10)

        self.controls_layout.addLayout(self.buttons_layout)

    def toggle_camera(self):
        if not self.thread.isRunning():
            self.camera_active = True
            self.thread._run_flag = True
            self.thread.start()

            self.btn_camera_toggle.setText("DETENER CÁMARA")
            self.btn_camera_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f; 
                    color: white; 
                    font-weight: bold; 
                    font-size: 14px; 
                    border-radius: 6px;
                }
                QPushButton:hover { background-color: #b71c1c; }
            """)
        else:
            self.camera_active = False
            self.thread.stop()
            self.image_label.clear()
            self.image_label.setText("Esperando cámara...")
            self.image_label.setStyleSheet("color: #666; font-size: 16px;")
            self.reset_checklist_ui()

            self.btn_camera_toggle.setText("INICIAR CÁMARA")
            self.btn_camera_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #007acc; 
                    color: white; 
                    font-weight: bold; 
                    font-size: 14px; 
                    border-radius: 6px;
                }
                QPushButton:hover { background-color: #0062a3; }
            """)

    def toggle_debug_view(self, checked):
        self.debug_mode = checked
        self.left_debug_container.setVisible(checked)

        if not checked:
            QTimer.singleShot(50, lambda: self.resize(self.width(), 0))

    def update_image(self, cv_img):
        if not self.camera_active and not self.showing_preview:
            return
        if self.showing_preview:
            return

        try:
            if cv_img is None: return
            use_gpu = self.chk_gpu.isChecked()
            alpha_calc = 1.0 + (self.contrast_value / 100.0)

            if use_gpu:
                umat_img = cv2.UMat(cv_img)
                processed_umat = cv2.convertScaleAbs(umat_img, alpha=alpha_calc, beta=self.brightness_value)
                processed_img = processed_umat.get()
            else:
                processed_img = cv2.convertScaleAbs(cv_img, alpha=alpha_calc, beta=self.brightness_value)

            self.frame_counter += 1
            should_analyze = True if use_gpu else (self.frame_counter % 3 == 0)

            if should_analyze:
                analyzed_img, status, metrics, is_ready, pts = self.analyzer.process(processed_img)
                self.last_results = (status, metrics, is_ready, pts)

                if self.debug_mode and metrics:
                    try:
                        # Formatear a espacios fijos ayuda a prevenir saltos si el número cambia bruscamente
                        self.lbl_yaw.setText(f"Yaw: {metrics.get('yaw', 0):.1f}°")
                        self.lbl_pitch.setText(f"Pitch: {metrics.get('pitch', 0):.1f}°")
                        self.lbl_roll.setText(f"Roll: {metrics.get('roll', 0):.1f}°")
                        self.lbl_blur.setText(f"Blur: {metrics.get('blur', 0):.1f}")

                        # Textos separados
                        self.lbl_light.setText(f"Luz: {metrics.get('brightness', 0):.1f}")

                        # En caso de que tengas 'area' en vez de 'lighting_dev'
                        if "area" in metrics:
                            self.lbl_light_dev.setText(f"Área: {metrics.get('area', 0):.1f}")
                        else:
                            self.lbl_light_dev.setText(f"Dev: {metrics.get('lighting_dev', 0):.1f}")

                        self.lbl_gaze.setText(f"Gaze: {metrics.get('gaze', 0):.3f}")

                        # Textos separados
                        self.lbl_mouth.setText(f"Boca: {metrics.get('mouth', 0):.3f}")
                        self.lbl_smile.setText(f"Sonrisa: {metrics.get('smile', 0):.3f}")
                    except:
                        pass
            else:
                if self.last_results:
                    status, metrics, is_ready, pts = self.last_results

                    if self.debug_mode and metrics:
                        try:
                            # Formatear a espacios fijos ayuda a prevenir saltos si el número cambia bruscamente
                            self.lbl_yaw.setText(f"Yaw: {metrics.get('yaw', 0):.1f}°")
                            self.lbl_pitch.setText(f"Pitch: {metrics.get('pitch', 0):.1f}°")
                            self.lbl_roll.setText(f"Roll: {metrics.get('roll', 0):.1f}°")
                            self.lbl_blur.setText(f"Blur: {metrics.get('blur', 0):.1f}")

                            # Textos separados
                            self.lbl_light.setText(f"Luz: {metrics.get('brightness', 0):.1f}")

                            # En caso de que tengas 'area' en vez de 'lighting_dev'
                            if "area" in metrics:
                                self.lbl_light_dev.setText(f"Área: {metrics.get('area', 0):.1f}")
                            else:
                                self.lbl_light_dev.setText(f"Dev: {metrics.get('lighting_dev', 0):.1f}")

                            self.lbl_gaze.setText(f"Gaze: {metrics.get('gaze', 0):.3f}")

                            # Textos separados
                            self.lbl_mouth.setText(f"Boca: {metrics.get('mouth', 0):.3f}")
                            self.lbl_smile.setText(f"Sonrisa: {metrics.get('smile', 0):.3f}")
                        except:
                            pass

                    analyzed_img = processed_img.copy()
                    h, w, _ = analyzed_img.shape

                    color = (76, 175, 80) if is_ready else (200, 200, 200)
                    thick = 3 if is_ready else 1
                    self.analyzer.draw_face_guide(analyzed_img, w, h, color, thick)

                    if pts is not None and len(pts) > 0:
                        import numpy as np
                        x1, y1 = np.min(pts, axis=0)
                        x2, y2 = np.max(pts, axis=0)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        color = (0, 255, 0) if is_ready else (0, 0, 255)
                        thickness = 2
                        line_len = 25

                        cv2.line(analyzed_img, (x1, y1), (x1 + line_len, y1), color, thickness)
                        cv2.line(analyzed_img, (x1, y1), (x1, y1 + line_len), color, thickness)

                        cv2.line(analyzed_img, (x2, y1), (x2 - line_len, y1), color, thickness)
                        cv2.line(analyzed_img, (x2, y1), (x2, y1 + line_len), color, thickness)

                        cv2.line(analyzed_img, (x1, y2), (x1 + line_len, y2), color, thickness)
                        cv2.line(analyzed_img, (x1, y2), (x1, y2 - line_len), color, thickness)

                        cv2.line(analyzed_img, (x2, y2), (x2 - line_len, y2), color, thickness)
                        cv2.line(analyzed_img, (x2, y2), (x2, y2 - line_len), color, thickness)

                else:
                    analyzed_img = processed_img
                    status, metrics, is_ready, pts = {}, {}, False, None

            self.update_checklist_ui(status)

            if self.is_auto_capturing:
                total_time = 3.0
                progress = (total_time - (self.countdown_value - 0.5)) / total_time
                # Aseguramos que el progreso no se pase de 1.0 por los decimales de tiempo
                progress = max(0.0, min(1.0, progress))

                # Alterna entre style=1 (Barra) y style=2 (Esquinas) para probar
                self.analyzer.draw_countdown_minimalist(
                    analyzed_img,
                    analyzed_img.shape[1],
                    analyzed_img.shape[0],
                    progress,
                    pts=pts,
                    style=1
                )

            if is_ready:
                self.current_processed_frame = processed_img
                self.last_valid_pts = pts
                self.btn_capture.setEnabled(True)
                self.start_auto_capture()
                self.btn_capture.setStyleSheet(
                    "background-color: #28a745; color: white; font-weight: bold; font-size: 14px; margin-top: 10px;")
            else:
                self.current_processed_frame = None
                self.btn_capture.setEnabled(False)
                self.cancel_auto_capture()
                self.btn_capture.setStyleSheet(
                    "background-color: #444; color: gray; font-weight: bold; font-size: 14px; margin-top: 10px;")

            if analyzed_img is not None:
                rgb_image = cv2.cvtColor(analyzed_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.image_label.setPixmap(
                    QPixmap.fromImage(qt_img.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)))

        except Exception as e:
            print(f"Error en el hilo de UI: {e}")

    def update_checklist_ui(self, status):
        face_detected = status.get("face_detected", False)

        for key, (lbl_icon, lbl_text, icon_name) in self.check_widgets.items():

            # Reglas inactivas (cuando no hay rostro)
            if key != "face_detected" and not face_detected:
                color = "#444444"  # Gris oscuro
                lbl_text.setStyleSheet(f"color: {color}; font-size: 13px;")
                lbl_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(20, 20))
                continue

            # Validación en tiempo real
            if status.get(key, False):
                # Validado
                color = "#4caf50"  # Verde éxito
                lbl_text.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
                lbl_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(20, 20))
            else:
                # Falló
                color = "#e57373"  # Rojo suave
                lbl_text.setStyleSheet(f"color: {color}; font-size: 13px;")
                lbl_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(20, 20))

    def reset_checklist_ui(self):
        for key, (lbl_icon, lbl_text, icon_name) in self.check_widgets.items():
            color = "#444444"
            lbl_text.setStyleSheet(f"color: {color}; font-size: 13px;")
            lbl_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(20, 20))

    def save_photo(self):
        if self.current_processed_frame is None: return

        try:
            self.auto_capture_timer.stop()
            self.is_auto_capturing = False
            self.showing_preview = True

            if self.thread.isRunning():
                self.thread.stop()

            self.btn_capture.setText("PROCESANDO...")
            self.btn_capture.repaint()
            QApplication.processEvents()

            self.preview_raw_img, self.preview_mask = self.image_processor.get_preview_data(
                self.current_processed_frame, self.last_valid_pts
            )

            if self.preview_raw_img is not None:
                self.btn_capture.setText(" DESCARTAR")
                self.btn_capture.setIcon(qta.icon('fa5s.trash-alt', color='white'))  # Icono de papelera
                self.btn_capture.setStyleSheet("""
                                QPushButton { 
                                    background-color: #dc3545; 
                                    color: white; 
                                    border-radius: 6px; 
                                    font-weight: bold; 
                                    border: none; 
                                    font-size: 13px; 
                                }
                                QPushButton:hover { background-color: #c82333; }
                            """)
                self.btn_capture.setEnabled(True)
                self.btn_save_final.setVisible(True)

                # --- MAGIA UX: Ocultar ruido visual ---
                self.btn_camera_toggle.setVisible(False)
                self.group_checklist.setVisible(False)
                self.group_advanced.setVisible(False)

                self.update_preview_display()
            else:
                print("Error: get_preview_data devolvió None")
                self.reset_session()

        except Exception as e:
            print(f"Error salvando la foto: {e}")
            import traceback
            traceback.print_exc()
            self.reset_session()

    def reset_capture_button(self):
        self.btn_capture.setText("CAPTURAR ID")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def toggle_performance_mode(self, checked):
        print(f"Cambiando modo. GPU Activa: {checked}")
        self.analyzer = FaceAnalyzer(use_gpu_mode=checked)

    def setup_advanced_controls(self):
        self.group_advanced = QGroupBox("OPCIONES AVANZADAS")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 20, 15, 15)

        self.chk_gpu = QCheckBox("Modo Alto Rendimiento (GPU)")
        self.chk_gpu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_gpu.setStyleSheet("""
            QCheckBox { color: #4dc4ff; font-weight: bold; }
            QCheckBox::indicator:checked { background-color: #4dc4ff; border-color: #4dc4ff; }
        """)
        self.chk_gpu.setToolTip("Activa detección de iris y validación frame a frame.")
        self.chk_gpu.toggled.connect(self.toggle_performance_mode)
        layout.addWidget(self.chk_gpu)

        self.chk_debug = QCheckBox("Modo Depuración / Calibración")
        self.chk_debug.setCursor(Qt.CursorShape.PointingHandCursor)
        # Estilo naranja suave
        self.chk_debug.setStyleSheet("""
            QCheckBox { color: #ffb74d; font-weight: bold; }
            QCheckBox::indicator:checked { background-color: #ffb74d; border-color: #ffb74d; }
        """)
        self.chk_debug.setToolTip("Muestra datos numéricos de yaw, pitch, luz, etc.")
        self.chk_debug.toggled.connect(self.toggle_debug_view)
        layout.addWidget(self.chk_debug)

        self.group_advanced.setLayout(layout)
        self.controls_layout.addWidget(self.group_advanced)

    def start_auto_capture(self):
        if not self.is_auto_capturing:
            self.is_auto_capturing = True
            self.countdown_value = 3  # Reiniciamos a 3 segundos
            self.auto_capture_timer.start()

            self.btn_capture.setText(f"✋ QUIETO... {self.countdown_value}")
            self.btn_capture.setStyleSheet("""
                background-color: #ff9800; 
                color: white; 
                font-weight: bold; 
                font-size: 16px; 
                border: 2px solid white;
            """)

    def cancel_auto_capture(self):
        if self.is_auto_capturing:
            self.is_auto_capturing = False
            self.auto_capture_timer.stop()
            self.countdown_value = 3
            self.btn_capture.setText("CAPTURAR ID")

    def tick_countdown(self):
        self.countdown_value -= 1

        if self.countdown_value <= 0:
            winsound.Beep(2000, 300)
            self.auto_capture_timer.stop()
            self.is_auto_capturing = False
            self.btn_capture.setText("¡CAPTURANDO!")
            self.btn_capture.setStyleSheet("background-color: #28a745; color: white;")
            self.save_photo()
        else:
            winsound.Beep(1000, 150) # Beep
            self.btn_capture.setText(f"✋ QUIETO... {self.countdown_value}")

    def handle_capture_click(self):
        if self.showing_preview:
            respuesta = QMessageBox.question(
                self,
                "Descartar Captura",
                "¿Estás seguro de que deseas descartar esta fotografía?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if respuesta == QMessageBox.StandardButton.Yes:
                self.reset_session()
        else:
            self.save_photo()

    def update_preview_display(self):
        if self.preview_raw_img is None or self.preview_mask is None or not self.showing_preview: return

        alpha = 1.0 + (self.contrast_value / 100.0)
        beta = self.brightness_value
        adjusted_face = cv2.convertScaleAbs(self.preview_raw_img, alpha=alpha, beta=beta)

        mask_f = self.preview_mask.astype(float)
        face_f = adjusted_face.astype(float)
        bg_white_f = (np.ones_like(adjusted_face) * 255).astype(float)

        final_view = (face_f * mask_f + bg_white_f * (1.0 - mask_f)).astype(np.uint8)

        rgb_preview = cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_preview.shape
        qt_img = QImage(rgb_preview.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

    def final_save_action(self):
        if self.preview_raw_img is None: return

        try:
            alpha = 1.0 + (self.contrast_value / 100.0)
            beta = self.brightness_value
            adjusted_face = cv2.convertScaleAbs(self.preview_raw_img, alpha=alpha, beta=beta)

            mask_f = self.preview_mask.astype(float)
            face_f = adjusted_face.astype(float)
            bg_white_f = (np.ones_like(adjusted_face) * 255).astype(float)
            final_img = (face_f * mask_f + bg_white_f * (1.0 - mask_f)).astype(np.uint8)

            filename = f"ID_FINAL_{int(time.time())}.jpg"
            cv2.imwrite(filename, final_img)
            print(f"Foto guardada: {filename}")

            QMessageBox.information(
                self,
                "Guardado Exitoso",
                f"¡El carnet se ha guardado correctamente!\n\nArchivo: {filename}"
            )

            self.reset_session()

        except Exception as e:
            print(f"Error al guardar: {e}")

    def reset_session(self):
        self.showing_preview = False
        self.is_auto_capturing = False
        self.countdown_value = 3
        self.preview_raw_img = None

        self.btn_save_final.setVisible(False)

        # Restaurar botón de captura
        self.btn_capture.setText(" CAPTURAR ID")
        self.btn_capture.setIcon(qta.icon('fa5s.camera', color='gray'))
        self.btn_capture.setEnabled(False)
        self.btn_capture.setStyleSheet(
            "background-color: #444; color: gray; font-weight: bold; font-size: 14px; margin-top: 10px;")

        # --- Mostrar de nuevo ---
        self.btn_camera_toggle.setVisible(True)
        self.group_checklist.setVisible(True)
        self.group_advanced.setVisible(True)

        if not self.thread.isRunning():
            self.thread._run_flag = True
            self.thread.start()

        self.reset_image_controls()