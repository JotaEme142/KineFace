import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque  # Para la memoria (buffer)


class KineFaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # --- ESTADOS ---
        self.is_calibrated = False
        self.reference_ratios = None

        # --- MEMORIA (MODO MANUAL) ---
        self.memory_size = 30  # 1 segundo a 30fps
        self.effort_buffer = deque(maxlen=self.memory_size)

        # --- PUNTOS Y MODELO DE POSE (SmartID heritage) ---
        self.POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nariz
            (0.0, -330.0, -65.0),  # Barbilla
            (-225.0, 170.0, -135.0),  # Ojo Izq
            (225.0, 170.0, -135.0),  # Ojo Der
            (-150.0, -150.0, -125.0),  # Boca Izq
            (150.0, -150.0, -125.0)  # Boca Der
        ])

        # Índices Ratios
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14]
        self.BROW_INNER_L, self.BROW_INNER_R = 107, 336
        self.EYE_INNER_L, self.EYE_INNER_R = 133, 362

    def get_head_pose(self, landmarks, w, h):
        """Versión normalizada para evitar el salto de 180 grados en Pitch"""
        img_pts = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in self.POSE_LANDMARKS],
                           dtype="double")
        focal_length = w
        center = (w / 2, h / 2)
        cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        # Resolvemos la pose
        _, rot_vec, _ = cv2.solvePnP(self.model_points, img_pts, cam_matrix, np.zeros((4, 1)),
                                     flags=cv2.SOLVEPNP_ITERATIVE)

        rmat, _ = cv2.Rodrigues(rot_vec)

        # Cálculo de ángulos
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2]) * 180 / np.pi
            yaw = np.arctan2(-rmat[2, 0], sy) * 180 / np.pi
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1]) * 180 / np.pi
            yaw = np.arctan2(-rmat[2, 0], sy) * 180 / np.pi

        # --- CORRECCIÓN DEL SALTO DE 180° (Normalización) ---
        # Si el sistema detecta que estás "al revés" (cerca de 180 o -180)
        # lo traemos al rango de 0 para que la validación funcione.
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch

        return pitch, yaw

    def validate_neutral_state(self, pts, p, y):
        """Verifica si el usuario está listo para calibrar con mayor tolerancia"""

        bar_val = self.get_bar(pts)
        ear_val = self.get_ear(pts)
        mar_val = self.get_mar(pts)

        # --- DEBUG: Mira estos valores en tu terminal mientras pruebas ---
        # print(f"P: {p:.1f} | Y: {y:.1f} | BAR: {bar_val:.2f} | EAR: {ear_val:.2f}")

        checks = {
            # Relajamos de 12 a 20 grados para Yaw y Pitch
            "frontal": abs(y) < 20 and abs(p) < 20,
            # Relajamos apertura de ojos (algunas personas tienen ojos más pequeños)
            "eyes_open": ear_val > 0.18,
            # Relajamos boca cerrada
            "mouth_closed": mar_val < 0.35,
            # BAR: Bajamos el umbral de 0.85 a 0.70 para permitir cejas naturalmente más juntas
            "neutral": bar_val > 0.70
        }

        return all(checks.values()), checks

    def get_ear(self, pts):
        def eye_dist(p): return (dist.euclidean(p[1], p[5]) + dist.euclidean(p[2], p[4])) / (
                    2.0 * dist.euclidean(p[0], p[3]))

        return (eye_dist(pts[self.LEFT_EYE]) + eye_dist(pts[self.RIGHT_EYE])) / 2.0

    def get_bar(self, pts):
        return dist.euclidean(pts[self.BROW_INNER_L], pts[self.BROW_INNER_R]) / (
                    dist.euclidean(pts[self.EYE_INNER_L], pts[self.EYE_INNER_R]) + 1e-6)

    def get_mar(self, pts):
        return dist.euclidean(pts[self.MOUTH[2]], pts[self.MOUTH[3]]) / (
                    dist.euclidean(pts[self.MOUTH[0]], pts[self.MOUTH[1]]) + 1e-6)

    def calibrate(self, pts):
        self.reference_ratios = {"ear": self.get_ear(pts), "bar": self.get_bar(pts), "mar": self.get_mar(pts)}
        self.is_calibrated = True

    def compute_manual_effort(self, pts):
        """Calcula el esfuerzo con 'Memoria' (Suavizado temporal)"""
        if not self.is_calibrated: return 0.0

        # Esfuerzo instantáneo
        d_bar = max(0, (self.reference_ratios["bar"] - self.get_bar(pts)) / self.reference_ratios["bar"])
        d_ear = max(0, (self.reference_ratios["ear"] - self.get_ear(pts)) / self.reference_ratios["ear"])
        d_mar = max(0, (self.get_mar(pts) - self.reference_ratios["mar"]) / (self.reference_ratios["mar"] + 1e-6))

        instant_score = (d_bar * 4.5) + (d_ear * 3.0) + (d_mar * 2.5)
        effort_val = np.clip(instant_score * 10, 0, 10)

        # --- APLICACIÓN DE MEMORIA (Suavizado) ---
        self.effort_buffer.append(effort_val)
        smoothed_effort = sum(self.effort_buffer) / len(self.effort_buffer)

        return round(smoothed_effort, 1)

    def process_frame(self, image):
        if image is None: return None
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return None
        lms = res.multi_face_landmarks[0].landmark
        pts = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in lms])
        p, y = self.get_head_pose(lms, w, h)
        return pts, p, y