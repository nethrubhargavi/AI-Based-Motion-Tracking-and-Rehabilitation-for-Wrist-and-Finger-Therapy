import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd
from mediapipe.framework.formats import landmark_pb2
from matplotlib.backends.backend_pdf import PdfPages
import os
import threading
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.session_id = None
        self.output_folder = None
        self.alpha = 0.7
        self.HAND_WIDTH_MM = 80
        self.MIN_HAND_WIDTH_PX = 40
        self.MIN_PX_PER_MM = 0.2
        self._reset_data()
        self.last_frame = None
        self.feedback_messages = [
            "Carefully Watch the Video",
            "Start the exersice and keep going",
            "Doing Great",
            "Good Work Keep Going",
            "Carefully Watch the Video",
            "Start the exersice, Your Doing Great Job",
            "Keep Going",
            "Your Almost there!, Excellent Work!",
            "Carefully Watch the Video",
            "Start the exersice",
            "You showed a Great progress",
            "Your Almost there, the session is coming to an End",
        ]
        self.current_feedback = "Starting session..."

    def _reset_data(self):
        self.time_data = []
        self.flexion_data = []
        self.extension_data = []
        self.radial_ulnar_data = []
        self.thumb_angles = {"CMC": [], "MCP": [], "IP": []}
        self.finger_angles = {
            "Index": {"PIP": [], "DIP": []},
            "Middle": {"PIP": [], "DIP": []},
            "Ring": {"PIP": [], "DIP": []},
            "Pinky": {"PIP": [], "DIP": []}
        }
        self.opposition_distances_mm = {"Index": [], "Middle": [], "Ring": [], "Pinky": []}
        self.fist_clench_metric_mm = []
        self.excel_time = []
        self.excel_angle = []
        self.excel_motion = []
        self.excel_finger_gesture = []
        self.prev_landmarks = None

    def _smooth_landmarks(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        smoothed = []
        for prev, curr in zip(self.prev_landmarks, current_landmarks):
            x = self.alpha * prev.x + (1 - self.alpha) * curr.x
            y = self.alpha * prev.y + (1 - self.alpha) * curr.y
            z = self.alpha * prev.z + (1 - self.alpha) * curr.z
            smoothed.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=z))
        self.prev_landmarks = smoothed
        return smoothed

    def get_feedback(self):
        if self.flexion_data and self.flexion_data[-1] > 45:
            self.current_feedback = "Perfect flexion angle! 💪"
        elif self.fist_clench_metric_mm and self.fist_clench_metric_mm[-1] < 20:
            self.current_feedback = "Great fist clench! ✊"
        else:
            self.current_feedback = np.random.choice(self.feedback_messages)
        return self.current_feedback

    def _compute_joint_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _compute_clinical_flexion(self, proximal, joint, distal):
        geometric_angle = self._compute_joint_angle(proximal, joint, distal)
        return 180 - geometric_angle

    def _validate_joint_angle(self, angle, joint_type):
        limits = {
            "DIP": 120, "PIP": 130, "IP": 120, "CMC": 50, "MCP": 55
        }
        return min(max(angle, 0), limits.get(joint_type, 180))

    def _compute_all_finger_angles(self, landmarks):
        lm = [(lm.x, lm.y, lm.z) for lm in landmarks]
        thumb = {
            "CMC": self._validate_joint_angle(self._compute_clinical_flexion(lm[0], lm[1], lm[2]), "CMC"),
            "MCP": self._validate_joint_angle(self._compute_clinical_flexion(lm[1], lm[2], lm[3]), "MCP"),
            "IP": self._validate_joint_angle(self._compute_clinical_flexion(lm[2], lm[3], lm[4]), "IP")
        }
        fingers = {
            "Index": {
                "PIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[5], lm[6], lm[7]), "PIP"),
                "DIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[6], lm[7], lm[8]), "DIP")
            },
            "Middle": {
                "PIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[9], lm[10], lm[11]), "PIP"),
                "DIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[10], lm[11], lm[12]), "DIP")
            },
            "Ring": {
                "PIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[13], lm[14], lm[15]), "PIP"),
                "DIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[14], lm[15], lm[16]), "DIP")
            },
            "Pinky": {
                "PIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[17], lm[18], lm[19]), "PIP"),
                "DIP": self._validate_joint_angle(self._compute_clinical_flexion(lm[18], lm[19], lm[20]), "DIP")
            }
        }
        return thumb, fingers

    def _get_flexion_extension(self, landmarks):
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        hand_direction = (index_mcp + pinky_mcp) / 2 - wrist
        hand_direction = hand_direction / np.linalg.norm(hand_direction)
        angle_rad = math.asin(hand_direction[1])
        angle_deg = math.degrees(angle_rad)
        if angle_deg > 10:
            motion = "Wrist Flexion"
        elif angle_deg < -20:
            motion = "Wrist Extension"
        else:
            motion = "Wrist Neutral"
        clamped_angle = min(abs(angle_deg), 90)
        return motion, clamped_angle

    def _get_radial_ulnar_motion(self, landmarks, hand_label):
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])
        pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])
        hand_center = (index_mcp + pinky_mcp) / 2
        movement_vector = hand_center - wrist
        if hand_label == "Right":
            if movement_vector[0] > 0.02:
                return "Radial Deviation"
            elif movement_vector[0] < -0.02:
                return "Ulnar Deviation"
        else:
            if movement_vector[0] < -0.02:
                return "Radial Deviation"
            elif movement_vector[0] > 0.02:
                return "Ulnar Deviation"
        return "Neutral"

    def _detect_finger_gesture(self, landmarks):
        lm = [(lm.x, lm.y, lm.z) for lm in landmarks]
        palm_base = lm[0][:2]
        thumb_tip = lm[4][:2]
        finger_tips = {
            "Index": lm[8][:2],
            "Middle": lm[12][:2],
            "Ring": lm[16][:2],
            "Pinky": lm[20][:2]
        }
        fingertips = [lm[i][:2] for i in [4,8,12,16,20]]
        dists_to_palm = [np.linalg.norm(np.array(f) - np.array(palm_base)) for f in fingertips]
        is_fist = all(d < 0.1 for d in dists_to_palm)
        finger_opposed = None
        for finger, tip_pos in finger_tips.items():
            dist = np.linalg.norm(np.array(thumb_tip) - np.array(tip_pos))
            if dist < 0.03:
                finger_opposed = finger
                break
        if is_fist:
            return "Fist Clenched"
        elif finger_opposed:
            return f"Thumb Opposing {finger_opposed}"
        else:
            return "Hand Open"

    def _get_hand_width_mm(self, landmarks, image_w, image_h):
        idx = [5, 17]
        pts = [landmarks[i] for i in idx]
        x1, y1 = pts[0].x * image_w, pts[0].y * image_h
        x2, y2 = pts[1].x * image_w, pts[1].y * image_h
        width_px = np.linalg.norm([x2 - x1, y2 - y1])
        if width_px < self.MIN_HAND_WIDTH_PX:
            return width_px, None
        px_per_mm = width_px / self.HAND_WIDTH_MM if self.HAND_WIDTH_MM > 0 else None
        if px_per_mm is not None and px_per_mm < self.MIN_PX_PER_MM:
            return width_px, None
        return width_px, px_per_mm

    def _compute_opposition_distances_mm(self, landmarks, image_w, image_h, px_per_mm):
        lm = [(lm.x * image_w, lm.y * image_h) for lm in landmarks]
        thumb_tip = lm[4]
        dists = {}
        for idx, finger in zip([8,12,16,20], ["Index", "Middle", "Ring", "Pinky"]):
            if px_per_mm and px_per_mm > self.MIN_PX_PER_MM:
                raw_px = np.linalg.norm(np.array(thumb_tip) - np.array(lm[idx]))
                raw_mm = raw_px / px_per_mm
                mm = 0.0 if raw_mm < 3.0 else raw_mm
                mm = mm if 0 <= mm < 150 else np.nan
            else:
                mm = np.nan
            dists[finger] = mm
        return dists

    def _compute_fist_clench_metric_mm(self, landmarks, image_w, image_h, px_per_mm):
        lm = [(lm.x * image_w, lm.y * image_h) for lm in landmarks]
        palm_base = lm[0]
        fingertips = [lm[i] for i in [4, 8, 12, 16, 20]]
        dists = []
        for f in fingertips:
            if px_per_mm and px_per_mm > self.MIN_PX_PER_MM:
                raw_mm = np.linalg.norm(np.array(f) - np.array(palm_base)) / px_per_mm
                mm = raw_mm if 0 <= raw_mm < 150 else np.nan
            else:
                mm = np.nan
            dists.append(mm)
        return np.nanmean(dists)

    def _generate_outputs(self, session_name):
        safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        self.output_folder = os.path.join("outputs", safe_name)
        os.makedirs(self.output_folder, exist_ok=True)
        excel_path = self._save_excel(safe_name)
        summary_pdf = self._save_summary_pdf(safe_name)
        plots_pdf = self._save_plots_pdf(safe_name)
        return {
            "excel": excel_path,
            "summary_pdf": summary_pdf,
            "plots_pdf": plots_pdf,
            "session_name": safe_name
        }

    def _save_excel(self, safe_name):
        excel_path = os.path.join(self.output_folder, f"{safe_name}.xlsx")
        df_dict = {
            "Time_s": self.time_data,
            "Wrist_Flexion_Angle": self.flexion_data,
            "Wrist_Extension_Angle": self.extension_data,
            "Radial_Ulnar_Deviation": self.radial_ulnar_data,
            "Wrist_Motion": self.excel_motion,
            "Finger_Gesture": self.excel_finger_gesture,
            "Fist_Clench_Metric_mm": self.fist_clench_metric_mm
        }
        for joint in ["CMC", "MCP", "IP"]:
            df_dict[f"Thumb_{joint}_Angle"] = self.thumb_angles[joint]
        for finger in ["Index", "Middle", "Ring", "Pinky"]:
            for joint in ["PIP", "DIP"]:
                df_dict[f"{finger}_{joint}_Angle"] = self.finger_angles[finger][joint]
            df_dict[f"Opposition_{finger}_mm"] = self.opposition_distances_mm[finger]
        df = pd.DataFrame(df_dict)
        df.to_excel(excel_path, index=False)
        return excel_path

    def _save_summary_pdf(self, safe_name):
        pdf_path = os.path.join(self.output_folder, f"{safe_name}_summary.pdf")
        excel_path = os.path.join(self.output_folder, f"{safe_name}.xlsx")
        df = pd.read_excel(excel_path)
        NORMAL_RANGES = {
            "Wrist Flexion": "0-80°",
            "Wrist Extension": "0-70°",
            "Thumb CMC": "0-55°",
            "Thumb MCP": "0-55°",
            "Thumb IP": "0-90°",
            "Index PIP": "0-115°",
            "Index DIP": "0-90°",
            "Middle PIP": "0-115°",
            "Middle DIP": "0-90°",
            "Ring PIP": "0-155°",
            "Ring DIP": "0-90°",
            "Pinky PIP": "0-115°",
            "Pinky DIP": "0-90°"
        }
        summary_data = [
            ("Wrist Flexion", df["Wrist_Flexion_Angle"].mean(skipna=True), df["Wrist_Flexion_Angle"].min(skipna=True), df["Wrist_Flexion_Angle"].max(skipna=True), NORMAL_RANGES["Wrist Flexion"]),
            ("Wrist Extension", df["Wrist_Extension_Angle"].mean(skipna=True), df["Wrist_Extension_Angle"].min(skipna=True), df["Wrist_Extension_Angle"].max(skipna=True), NORMAL_RANGES["Wrist Extension"])
        ]
        for joint in ["CMC", "MCP", "IP"]:
            key = f"Thumb_{joint}_Angle"
            label = f"Thumb {joint}"
            summary_data.append((label, df[key].mean(skipna=True), df[key].min(skipna=True), df[key].max(skipna=True), NORMAL_RANGES[label]))
        for finger in ["Index", "Middle", "Ring", "Pinky"]:
            for joint in ["PIP", "DIP"]:
                key = f"{finger}_{joint}_Angle"
                label = f"{finger} {joint}"
                summary_data.append((label, df[key].mean(skipna=True), df[key].min(skipna=True), df[key].max(skipna=True), NORMAL_RANGES[label]))
        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Mean Value", "Min Value", "Max Value", "Normal Range"])
        fig, ax = plt.subplots(figsize=(10, len(summary_df)*0.5+2))
        ax.axis('off')
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(summary_df.columns))))
        plt.title("Session Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(pdf_path)
        plt.close(fig)
        return pdf_path

    def _save_plots_pdf(self, safe_name):
        pdf_path = os.path.join(self.output_folder, f"{safe_name}_plots.pdf")
        with PdfPages(pdf_path) as pdf:
            # Flexion/Extension
            fig, ax = plt.subplots()
            ax.plot(self.time_data, self.flexion_data, label='Flexion')
            ax.plot(self.time_data, self.extension_data, label='Extension')
            ax.set_title("Wrist Flexion & Extension")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle (°)")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
            # Thumb Angles
            fig, ax = plt.subplots()
            for joint, color in zip(["CMC", "MCP", "IP"], ["red", "green", "blue"]):
                ax.plot(self.time_data, self.thumb_angles[joint], label=f"Thumb {joint}", color=color)
            ax.set_title("Thumb Joint Angles")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle (°)")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
            # Finger Angles
            for joint in ["PIP", "DIP"]:
                fig, ax = plt.subplots()
                for finger, color in zip(["Index", "Middle", "Ring", "Pinky"], ["blue", "green", "orange", "purple"]):
                    ax.plot(self.time_data, self.finger_angles[finger][joint], label=f"{finger} {joint}", color=color)
                ax.set_title(f"Finger {joint} Angles")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Angle (°)")
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)
            # Opposition Distances
            fig, ax = plt.subplots()
            for finger, color in zip(["Index", "Middle", "Ring", "Pinky"], ["blue", "green", "orange", "purple"]):
                ax.plot(self.time_data, self.opposition_distances_mm[finger], label=f"{finger}", color=color)
            ax.set_title("Thumb-Finger Opposition Distances")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance (mm)")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
            # Fist Clench Metric
            fig, ax = plt.subplots()
            ax.plot(self.time_data, self.fist_clench_metric_mm, label="Fist Clench Metric", color="black")
            ax.set_title("Fist Clench Metric")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Distance (mm)")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        return pdf_path

    def raw_stream(self):
        self._reset_data()
        self.stop_event.clear()
        cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            success, image = cap.read()
            if not success:
                continue
            self.last_frame = image.copy()
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    def processed_stream(self):
        self._reset_data()
        self.stop_event.clear()
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        ) as hands:
            while not self.stop_event.is_set():
                success, image = cap.read()
                if not success:
                    continue
                image = cv2.resize(image, (800, 600))
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                elapsed_time = time.time() - start_time
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_label = results.multi_handedness[0].classification[0].label
                    smoothed_landmarks = self._smooth_landmarks(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist_motion, angle = self._get_flexion_extension(smoothed_landmarks)
                    deviation = self._get_radial_ulnar_motion(smoothed_landmarks, hand_label)
                    finger_gesture = self._detect_finger_gesture(smoothed_landmarks)
                    hand_width_px, px_per_mm = self._get_hand_width_mm(smoothed_landmarks, w, h)
                    thumb, fingers = self._compute_all_finger_angles(smoothed_landmarks)
                    for joint in ["CMC", "MCP", "IP"]:
                        self.thumb_angles[joint].append(thumb[joint])
                    for finger in ["Index", "Middle", "Ring", "Pinky"]:
                        for joint in ["PIP", "DIP"]:
                            self.finger_angles[finger][joint].append(fingers[finger][joint])
                        self.opposition_distances_mm[finger].append(
                            self._compute_opposition_distances_mm(smoothed_landmarks, w, h, px_per_mm)[finger]
                        )
                    self.fist_clench_metric_mm.append(
                        self._compute_fist_clench_metric_mm(smoothed_landmarks, w, h, px_per_mm)
                    )
                    self.time_data.append(round(elapsed_time, 2))
                    self.flexion_data.append(angle if "Flexion" in wrist_motion else 0)
                    self.extension_data.append(angle if "Extension" in wrist_motion else 0)
                    self.radial_ulnar_data.append(deviation)
                    self.excel_time.append(round(elapsed_time, 2))
                    self.excel_angle.append(round(angle, 2))
                    self.excel_motion.append(wrist_motion)
                    self.excel_finger_gesture.append(finger_gesture)
                    text = f'{hand_label}: {wrist_motion} + {deviation} | {finger_gesture}'
                    wrist = smoothed_landmarks[0]
                    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                    color = (0, 255, 0) if angle < 70 else (0, 0, 255)
                    cv2.putText(image, text, (wrist_coords[0] - 300, wrist_coords[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    def stop_tracking(self, session_name="session"):
        self.stop_event.set()
        return self._generate_outputs(session_name)
