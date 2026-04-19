import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist


class FaceAnalyzer:
    def __init__(self, use_gpu_mode=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        # SI ES CPU: Apagamos refine_landmarks para ganar velocidad (modelo ligero)
        # SI ES GPU: Lo encendemos para máxima precisión (modelo pesado)
        self.refine_iris = True
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=self.refine_iris,  # Esto activa los puntos del iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Detector de manos
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            model_complexity=0
        )

        # Variables para optimización
        self.frame_count = 0
        self.last_hands_detected = False
        self.cached_hands_landmarks = None

        # Umbrales calibrados
        self.TH_BLUR = 3.0
        self.TH_EAR = 0.20
        self.TH_GAZE = 0.08
        self.TH_MOUTH = 0.30
        self.TH_SMILE = 0.43
        self.TH_FROWN = 0.80
        self.TH_EYEBROW = 0.16
        self.TH_BG_STD = 60.0
        self.TH_OCCLUSION = 50.0
        self.TH_LIGHT_UNIFORM = 40.0
        self.TH_AREA_MIN = 0.12
        self.TH_AREA_MAX = 0.22

        self.YAW_LIMIT = 15
        self.PITCH_LIMIT = 15
        self.ROLL_LIMIT = 12

        self.guide_img = None
        self._guide_cache = {
            "last_w": 0, "last_h": 0, 
            "mask": None,
            "inv_mask": None,
            "tinted_green": None,
            "tinted_red": None,
            "tinted_white": None
        }
        
        try:
            import os
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            asset_path = os.path.join(base_path, "assets", "guide.png")
            if os.path.exists(asset_path):
                self.guide_img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error cargando guía: {e}")

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14]

        self.POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

    def set_thresholds(self, thresholds: dict):
        if "blur" in thresholds: self.TH_BLUR = thresholds["blur"]
        if "ear" in thresholds: self.TH_EAR = thresholds["ear"]
        if "gaze" in thresholds: self.TH_GAZE = thresholds["gaze"]
        if "mouth" in thresholds: self.TH_MOUTH = thresholds["mouth"]
        if "smile" in thresholds: self.TH_SMILE = thresholds["smile"]
        if "frown" in thresholds: self.TH_FROWN = thresholds["frown"]
        if "eyebrow" in thresholds: self.TH_EYEBROW = thresholds["eyebrow"]
        if "bg" in thresholds: self.TH_BG_STD = thresholds["bg"]
        if "occlusion" in thresholds: self.TH_OCCLUSION = thresholds["occlusion"]
        if "dist_min" in thresholds: self.TH_AREA_MIN = thresholds["dist_min"]
        if "dist_max" in thresholds: self.TH_AREA_MAX = thresholds["dist_max"]
        if "yaw" in thresholds: self.YAW_LIMIT = thresholds["yaw"]
        if "pitch" in thresholds: self.PITCH_LIMIT = thresholds["pitch"]
        if "roll" in thresholds: self.ROLL_LIMIT = thresholds["roll"]

    def process(self, image):
        if image is None: return None, {}, {}, False, None
        h, w, _ = image.shape
        img_out = image.copy()

        self.frame_count += 1

        status = {
            "face_detected": False, "centered": False, "distance": False,
            "pose": False, "gaze": False, "eyes": False, "mouth": False,
            "no_hands": True, "no_occlusion": False, "neutral_exp": False,
            "lighting": False, "lighting_uniform": False, "sharpness": False, "background_ok": False
        }
        metrics = {"blur": 0, "yaw": 0, "pitch": 0, "roll": 0, "brightness": 0, "lighting_dev": 0, "gaze": 0, "mouth": 0}

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.frame_count % 5 == 0:
                hands_res = self.hands_detector.process(rgb) # Procesar manos
                self.last_hands_detected = False
                self.cached_hands_landmarks = hands_res.multi_hand_landmarks

            res = self.face_mesh.process(rgb)
            lms_list = getattr(res, 'multi_face_landmarks', None)

            if not lms_list:
                self.draw_face_guide(img_out, w, h, (100, 100, 100), 1)
                return img_out, status, metrics, False, None

            lms = lms_list[0].landmark
            if len(lms) < 468: return img_out, status, metrics, False, None

            status["face_detected"] = True
            pts = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in lms])

            # Coordenadas Bbox
            x1, y1 = np.clip(np.min(pts, axis=0), 0, [w, h])
            x2, y2 = np.clip(np.max(pts, axis=0), 0, [w, h])

            # Validar manos
            hands_on_face = False
            if hasattr(self, 'cached_hands_landmarks') and self.cached_hands_landmarks:
                for hand_lms in self.cached_hands_landmarks:
                    for h_lm in hand_lms.landmark:
                        hx, hy = int(h_lm.x * w), int(h_lm.y * h)
                        pad = 30
                        if (x1 - pad) < hx < (x2 + pad) and (y1 - pad) < hy < (y2 + pad):
                            hands_on_face = True
                            break
                    if hands_on_face: break

            if hands_on_face:
                status["no_hands"] = False
            else:
                status["no_hands"] = True

            # Verificar oclusión
            try:
                ref_indices = [10, 234, 454]
                ref_chroma = []

                for idx in ref_indices:
                    px, py = pts[idx]
                    region = image[py - 2:py + 3, px - 2:px + 3]
                    if region.size > 0:
                        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
                        mean_color = np.mean(ycrcb, axis=(0, 1))
                        ref_chroma.append(mean_color[1:])

                if ref_chroma:
                    skin_base = np.mean(ref_chroma, axis=0)
                else:
                    skin_base = np.array([0, 0])

                risk_indices = [1, 13, 152]
                max_diff = 0

                for idx in risk_indices:
                    px, py = pts[idx]
                    region = image[py - 2:py + 3, px - 2:px + 3]
                    if region.size > 0:
                        ycrcb_risk = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
                        risk_chroma = np.mean(ycrcb_risk, axis=(0, 1))[1:]

                        diff = np.linalg.norm(skin_base - risk_chroma)
                        if diff > max_diff:
                            max_diff = diff

                metrics["occlusion_diff"] = int(max_diff)
                if max_diff < (self.TH_OCCLUSION * 0.4):
                    status["no_occlusion"] = True
                else:
                    status["no_occlusion"] = False

            except Exception as e:
                print(f"Error Oclusión: {e}")
                status["no_occlusion"] = True

            # Centrado
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if abs(cx - w // 2) < w * 0.06 and abs(cy - h * 0.35) < h * 0.08:
                status["centered"] = True

            # Distancia
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            metrics["area"] = round(area, 3)
            if self.TH_AREA_MIN < area < self.TH_AREA_MAX:
                status["distance"] = True

            # Postura y Ojos
            p, y, r = self.get_head_pose(lms, w, h)
            metrics.update({"yaw": int(y), "pitch": int(p), "roll": int(r)})
            if abs(y) < 15 and abs(p) < 15 and abs(r) < 12: status["pose"] = True

            l_ear, r_ear = self.ear(pts[self.LEFT_EYE]), self.ear(pts[self.RIGHT_EYE])
            if (l_ear + r_ear) / 2 > self.TH_EAR:
                status["eyes"] = True
                if len(lms) > 470:
                    g_ok, g_off = self.check_gaze(lms)
                    status["gaze"] = g_ok
                    metrics["gaze"] = g_off
                else:
                    status["gaze"] = False
                    metrics["gaze"] = 0
            else:
                # Si los ojos están cerrados, por lógica no puede estar mirando al frente
                status["eyes"] = False
                status["gaze"] = False
                metrics["gaze"] = 0

            # Boca Cerrada
            m_ratio = self.mouth_ratio(pts[self.MOUTH])
            metrics["mouth"] = round(m_ratio, 3)
            if m_ratio < self.TH_MOUTH: status["mouth"] = True

            # Expresión Neutra (Análisis Morfológico)
            # SONRISA (Expansión horizontal)
            d_mouth_w = dist.euclidean(pts[61], pts[291])
            d_face_w = dist.euclidean(pts[234], pts[454])
            smile_ratio = d_mouth_w / (d_face_w + 1e-6)

            # CEÑO FRUNCIDO (Contracción de cejas internas)
            d_inner_brows = dist.euclidean(pts[107], pts[336])
            d_eyes_inner = dist.euclidean(pts[133], pts[362])
            frown_ratio = d_inner_brows / (d_eyes_inner + 1e-6)

            # CEJAS LEVANTADAS (Expansión vertical superior)
            d_brow_eye_l = dist.euclidean(pts[105], pts[159])
            d_brow_eye_r = dist.euclidean(pts[334], pts[386])
            avg_brow_h = (d_brow_eye_l + d_brow_eye_r) / 2.0
            d_face_h = dist.euclidean(pts[10], pts[152])
            eyebrow_ratio = avg_brow_h / (d_face_h + 1e-6)

            # Guardamos para modo debug
            metrics["smile"] = round(smile_ratio, 3)
            metrics["frown"] = round(frown_ratio, 3)
            metrics["eyebrow"] = round(eyebrow_ratio, 3)

            if (smile_ratio < self.TH_SMILE) and (frown_ratio > self.TH_FROWN) and (eyebrow_ratio < self.TH_EYEBROW):
                status["neutral_exp"] = True
            else:
                status["neutral_exp"] = False

            # Calidad ROI y Fondo
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
                total_pixels = gray_roi.size

                dark_pixels = np.sum(hist[:10])
                bright_pixels = np.sum(hist[250:])

                dark_ratio = dark_pixels / total_pixels
                bright_ratio = bright_pixels / total_pixels

                metrics["brightness"] = int(np.mean(gray_roi))
                if dark_ratio < 0.15 and bright_ratio < 0.15 and 60 < metrics["brightness"] < 220:
                    status["lighting"] = True
                else:
                    status["lighting"] = False

                h_roi, w_roi = gray_roi.shape
                left_half = gray_roi[:, :w_roi // 2]
                right_half = gray_roi[:, w_roi // 2:]

                l_mean = np.mean(left_half)
                r_mean = np.mean(right_half)
                diff_light = abs(l_mean - r_mean)

                metrics["lighting_dev"] = int(diff_light)

                if diff_light < self.TH_LIGHT_UNIFORM:
                    status["lighting_uniform"] = True
                else:
                    status["lighting_uniform"] = False

                blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
                laplacian_var = cv2.Laplacian(blurred_roi, cv2.CV_64F).var()

                metrics["blur"] = int(laplacian_var)

                if metrics["blur"] > self.TH_BLUR:
                    status["sharpness"] = True

                # Validación de fondo simple
                bg_sample_l = image[10:50, 10:50]
                bg_sample_r = image[10:50, w - 50:w - 10]
                std_l = np.std(bg_sample_l)
                std_r = np.std(bg_sample_r)
                avg_std = (std_l + std_r) / 2
                metrics["bg_std"] = int(avg_std)
                if avg_std < self.TH_BG_STD: status["background_ok"] = True

            is_ready = all(status.values())

            # Dibujado
            overlay_color = (76, 175, 80) if is_ready else (200, 200, 200)
            overlay_thick = 3 if is_ready else 1
            
            self.draw_face_guide(img_out, w, h, overlay_color, overlay_thick)

            color = (76, 175, 80) if is_ready else (0, 0, 255)
            thickness = 2
            line_len = 30

            off = 0 if is_ready else int(np.sin(time.time() * 10) * 3)
            x1, y1, x2, y2 = x1-off, y1-off, x2+off, y2+off

            cv2.line(img_out, (x1, y1), (x1 + line_len, y1), color, thickness)
            cv2.line(img_out, (x1, y1), (x1, y1 + line_len), color, thickness)

            cv2.line(img_out, (x2, y1), (x2 - line_len, y1), color, thickness)
            cv2.line(img_out, (x2, y1), (x2, y1 + line_len), color, thickness)

            cv2.line(img_out, (x1, y2), (x1 + line_len, y2), color, thickness)
            cv2.line(img_out, (x1, y2), (x1, y2 - line_len), color, thickness)

            cv2.line(img_out, (x2, y2), (x2 - line_len, y2), color, thickness)
            cv2.line(img_out, (x2, y2), (x2, y2 - line_len), color, thickness)

            return img_out, status, metrics, is_ready, pts

        except Exception as e:
            print(f"Crash prevenido: {e}")
            return img_out, status, metrics, False, None

    def get_head_pose(self, landmarks, w, h):
        img_pts = []
        for idx in self.POSE_LANDMARKS:
            lm = landmarks[idx]
            img_pts.append((lm.x * w, lm.y * h))
        img_pts = np.array(img_pts, dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        pitch = x * 180.0 / np.pi
        yaw = y * 180.0 / np.pi
        roll = z * 180.0 / np.pi

        # Pitch: Mirar arriba/abajo.
        if pitch > 0:
            pitch = 180 - pitch
        else:
            pitch = -180 - pitch
        return pitch, yaw, roll

    def ear(self, pts):
        return (dist.euclidean(pts[1], pts[5]) + dist.euclidean(pts[2], pts[4])) / (
                    2.0 * dist.euclidean(pts[0], pts[3]))

    def mouth_ratio(self, pts):
        d_v = dist.euclidean(pts[2], pts[3])
        d_h = dist.euclidean(pts[0], pts[1])
        return d_v / (d_h + 1e-6)

    def draw_face_guide(self, img, w, h, color, thickness):
        cx, cy = w // 2, int(h * 0.35)
        base = min(w, h)
        rx, ry = int(base * 0.18), int(base * 0.18 * 1.4)
        
        guide = self.guide_img
        if guide is not None:
            try:
                gh_raw, gw_raw = guide.shape[:2]
                asset_aspect = gw_raw / gh_raw
                render_h = int(h * 0.85)
                render_w = int(render_h * asset_aspect)
                if render_w > w * 0.9:
                    render_w = int(w * 0.9)
                    render_h = int(render_w / asset_aspect)

                cache = self._guide_cache
                if cache["last_w"] != render_w or cache["last_h"] != render_h:
                    resized = cv2.resize(guide, (render_w, render_h), interpolation=cv2.INTER_AREA)
                    if resized.shape[2] == 4:
                        _, _, _, a = cv2.split(resized)
                        _, mask_bin = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)

                        bg_black = np.zeros((render_h, render_w, 3), dtype=np.uint8)

                        cache["tinted_green"] = cv2.bitwise_and(np.full_like(bg_black, (76, 175, 80)), np.full_like(bg_black, (255,255,255)), mask=mask_bin)
                        cache["tinted_red"] = cv2.bitwise_and(np.full_like(bg_black, (50, 50, 200)), np.full_like(bg_black, (255,255,255)), mask=mask_bin)
                        cache["tinted_white"] = cv2.bitwise_and(resized[:,:,:3], resized[:,:,:3], mask=mask_bin)
                        
                        cache["mask"] = mask_bin
                        cache["inv_mask"] = cv2.bitwise_not(mask_bin)
                    else:
                        cache["tinted_white"] = resized
                        cache["mask"] = None
                    
                    cache["last_w"], cache["last_h"] = render_w, render_h

                if color[1] > color[2]:
                    asset_to_draw = cache["tinted_green"]
                elif color[2] > color[1]:
                    asset_to_draw = cache["tinted_red"]
                else:
                    asset_to_draw = cache["tinted_white"]

                head_ratio = 0.35
                y1, x1 = cy - int(render_h * head_ratio), cx - render_w // 2
                y1_s, y2_s = max(0, y1), min(h, y1 + render_h)
                x1_s, x2_s = max(0, x1), min(w, x1 + render_w)
                
                if y2_s > y1_s and x2_s > x1_s:
                    ay1, ax1 = y1_s - y1, x1_s - x1
                    ay2, ax2 = ay1 + (y2_s - y1_s), ax1 + (x2_s - x1_s)
                    
                    roi = img[y1_s:y2_s, x1_s:x2_s]
                    if cache["mask"] is not None:
                        m_inv = cache["inv_mask"][ay1:ay2, ax1:ax2]
                        a_roi = asset_to_draw[ay1:ay2, ax1:ax2]
                        roi_bg = cv2.bitwise_and(roi, roi, mask=m_inv)
                        img[y1_s:y2_s, x1_s:x2_s] = cv2.add(roi_bg, a_roi)
                    else:
                        cv2.addWeighted(roi, 0.5, asset_to_draw[ay1:ay2, ax1:ax2], 0.5, 0, dst=roi)
                    return
            except Exception as e:
                print(f"Error renderizando guía: {e}")

        sc_len = int(base * 0.04)

        cv2.line(img, (cx - rx, cy - ry), (cx - rx + sc_len, cy - ry), color, thickness + 1)
        cv2.line(img, (cx - rx, cy - ry), (cx - rx, cy - ry + sc_len), color, thickness + 1)

        cv2.line(img, (cx + rx, cy - ry), (cx + rx - sc_len, cy - ry), color, thickness + 1)
        cv2.line(img, (cx + rx, cy - ry), (cx + rx, cy - ry + sc_len), color, thickness + 1)

        cv2.line(img, (cx - rx, cy + ry), (cx - rx + sc_len, cy + ry), color, thickness + 1)
        cv2.line(img, (cx - rx, cy + ry), (cx - rx, cy + ry - sc_len), color, thickness + 1)

        cv2.line(img, (cx + rx, cy + ry), (cx + rx - sc_len, cy + ry), color, thickness + 1)
        cv2.line(img, (cx + rx, cy + ry), (cx + rx, cy + ry - sc_len), color, thickness + 1)

        sh_y = int(cy + ry * 1.5)
        sh_x_ext = int(rx * 1.8)
        cv2.line(img, (cx - sh_x_ext, sh_y), (cx - rx, sh_y), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx + rx, sh_y), (cx + sh_x_ext, sh_y), color, 1, cv2.LINE_AA)

        cv2.line(img, (cx - rx - 10, cy), (cx - rx, cy), color, 1)
        cv2.line(img, (cx + rx, cy), (cx + rx + 10, cy), color, 1)

    def draw_countdown_minimalist(self, img, w, h, progress, pts=None, style=1):
        if style == 1:
            # ESTILO 1: Barra de progreso flotante e inferior (Estilo iOS/Android)
            bar_w = int(w * 0.5)  # Ocupa el 50% del ancho de la pantalla
            bar_h = 6
            x = (w - bar_w) // 2
            y = h - 50  # Un poco separada del borde inferior

            # Fondo de la barra (Gris oscuro)
            cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1, cv2.LINE_AA)
            # Relleno de progreso (Verde)
            cv2.rectangle(img, (x, y), (x + int(bar_w * progress), y + bar_h), (76, 175, 80), -1, cv2.LINE_AA)

        elif style == 2:
            # ESTILO 2: Animación de las esquinas del rostro cerrándose
            if pts is not None and len(pts) > 0:
                x1, y1 = np.clip(np.min(pts, axis=0), 0, [w, h])
                x2, y2 = np.clip(np.max(pts, axis=0), 0, [w, h])

                box_w = x2 - x1
                box_h = y2 - y1

                # Longitud máxima necesaria para que las líneas se toquen en el centro
                max_len_x = box_w // 2
                max_len_y = box_h // 2

                # La longitud arranca en 30px (como tus brackets originales) y crece
                base_len = 30
                current_len_x = int(base_len + (max_len_x - base_len) * progress)
                current_len_y = int(base_len + (max_len_y - base_len) * progress)

                color = (76, 175, 80)
                thickness = 4  # Más grueso que el original para que destaque al animarse

                # Esquina Superior Izquierda
                cv2.line(img, (x1, y1), (x1 + current_len_x, y1), color, thickness)
                cv2.line(img, (x1, y1), (x1, y1 + current_len_y), color, thickness)
                # Esquina Superior Derecha
                cv2.line(img, (x2, y1), (x2 - current_len_x, y1), color, thickness)
                cv2.line(img, (x2, y1), (x2, y1 + current_len_y), color, thickness)
                # Esquina Inferior Izquierda
                cv2.line(img, (x1, y2), (x1 + current_len_x, y2), color, thickness)
                cv2.line(img, (x1, y2), (x1, y2 - current_len_y), color, thickness)
                # Esquina Inferior Derecha
                cv2.line(img, (x2, y2), (x2 - current_len_x, y2), color, thickness)
                cv2.line(img, (x2, y2), (x2, y2 - current_len_y), color, thickness)

    def check_gaze(self, lms):
        def get_off(iris, l, r):
            c = (lms[l].x + lms[r].x)/2
            w = abs(lms[r].x - lms[l].x)
            return (lms[iris].x - c)/(w + 1e-6)
        off = (get_off(468, 33, 133) + get_off(473, 362, 263))/2
        return abs(off) < self.TH_GAZE, abs(off)