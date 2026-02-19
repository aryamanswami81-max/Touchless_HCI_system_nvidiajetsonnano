import ctypes 

try: 

    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL) 

except: 

    pass 

 

import os 

import subprocess 

os.environ["QT_X11_NO_MITSHM"] = "1" 

 

import mediapipe as mp 

import cv2 

import time 

import numpy as np 

import joblib 

from collections import deque 

from threading import Thread, Lock 

from pynput.keyboard import Controller, Key 

 

model = joblib.load("hand_gesture.joblib") 

keyboard = Controller() 

 

HDMI_SINK = "alsa_output.platform-70030000.hda.hdmi-stereo" 

 

def volume_up():  

    subprocess.run(["amixer", "-c", "0", "sset", "IEC958", "5%+"], stdout=subprocess.DEVNULL) 

 

def volume_down():  

    subprocess.run(["amixer", "-c", "0", "sset", "IEC958", "5%-"], stdout=subprocess.DEVNULL) 

 

def volume_mute():  

    subprocess.run(["amixer", "-c", "0", "sset", "IEC958", "toggle"], stdout=subprocess.DEVNULL) 

 

def play_pause():  

    keyboard.press(Key.space) 

    keyboard.release(Key.space) 

 

def fast_forward():  

    keyboard.press(Key.right) 

    keyboard.release(Key.right) 

 

def rewind():  

    keyboard.press(Key.left) 

    keyboard.release(Key.left) 

 

mp_hands = mp.solutions.hands 

hands = mp_hands.Hands( 

    static_image_mode=False, 

    max_num_hands=1, 

    min_detection_confidence=0.6, 

    min_tracking_confidence=0.6 

) 

 

class CameraThread: 

    def __init__(self, src=0): 

        self.cap = cv2.VideoCapture(src) 

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

        self.frame = None 

        self.lock = Lock() 

        self.running = True 

        Thread(target=self.update, daemon=True).start() 

 

    def update(self): 

        while self.running: 

            ret, frame = self.cap.read() 

            if ret: 

                with self.lock: 

                    self.frame = frame 

 

    def read(self): 

        with self.lock: 

            return None if self.frame is None else self.frame.copy() 

 

    def stop(self): 

        self.running = False 

        self.cap.release() 

 

MODES = ["playback", "volume"] 

current_mode_index = 0 

mode_switch_time = 0 

 

prediction_history = deque(maxlen=3) 

gesture_hold_start = None 

current_active_gesture = None 

 

HOLD_TIME = 1.0 

CONF_THRESHOLD = 0.90 

 

camera = CameraThread(0) 

frame_counter = 0 

 

while True: 

    frame_start = time.perf_counter() 

    original_frame = camera.read() 

     

    if original_frame is None: 

        continue 

 

    frame_counter += 1 

    if frame_counter % 2 != 0: 

        continue 

 

    small = cv2.resize(original_frame, (256, 192)) 

    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB) 

 

    results = hands.process(frame_rgb) 

 

    h, w = original_frame.shape[:2] 

    detected_gesture = "none" 

    current_time = time.perf_counter() 

 

    if results.multi_hand_landmarks: 

        landmarks = results.multi_hand_landmarks[0].landmark 

        lm = np.zeros((21, 3), dtype=np.float32) 

        for i in range(21): 

            p = landmarks[i] 

            lm[i] = [p.x, p.y, p.z] 

 

        lm -= lm[0] 

        X_live = lm.reshape(1, -1) 

 

        probabilities = model.predict_proba(X_live)[0] 

        max_idx = np.argmax(probabilities) 

 

        if probabilities[max_idx] >= CONF_THRESHOLD: 

            detected_gesture = model.classes_[max_idx] 

 

    prediction_history.append(detected_gesture) 

    stable_gesture = max(set(prediction_history), key=prediction_history.count) 

 

    if stable_gesture != "none": 

        if stable_gesture != current_active_gesture: 

            gesture_hold_start = current_time 

            current_active_gesture = stable_gesture 

 

        elapsed = current_time - gesture_hold_start 

        progress = min(elapsed / HOLD_TIME, 1.0) 

 

        bar_w = int(w * 0.6) 

        bar_x = int((w - bar_w) / 2) 

 

        cv2.rectangle(original_frame, (bar_x, h-40), (bar_x + bar_w, h-25), (100, 100, 100), -1) 

        cv2.rectangle(original_frame, (bar_x, h-40), (bar_x + int(bar_w * progress), h-25), (0, 255, 0), -1) 

        cv2.putText(original_frame, stable_gesture.upper(), (bar_x, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 

 

        if progress >= 1.0: 

            if stable_gesture == "like": 

                current_mode_index = (current_mode_index + 1) % len(MODES) 

                mode_switch_time = current_time 

                prediction_history.clear() 

            else: 

                mode = MODES[current_mode_index] 

                if mode == "volume": 

                    if stable_gesture == "fist": volume_up() 

                    elif stable_gesture == "ok": volume_down() 

                    elif stable_gesture == "rock": volume_mute() 

                elif mode == "playback": 

                    if stable_gesture == "fist": play_pause() 

                    elif stable_gesture == "ok": fast_forward() 

                    elif stable_gesture == "rock": rewind() 

 

                prediction_history.clear() 

            gesture_hold_start = current_time 

    else: 

        gesture_hold_start = None 

        current_active_gesture = None 

 

    frame_end = time.perf_counter() 

    latency_ms = (frame_end - frame_start) * 1000 

    fps = 1.0 / (frame_end - frame_start) 

 

    cv2.rectangle(original_frame, (0, 0), (w, 60), (30, 30, 30), -1) 

    mode_text = "MODE: " + MODES[current_mode_index].upper() 

    mode_color = (0, 200, 255) if MODES[current_mode_index] == "playback" else (0, 255, 0) 

 

    cv2.putText(original_frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3) 

 

    if current_time - mode_switch_time < 1.0: 

        cv2.putText(original_frame, "MODE CHANGED", (w//2 - 180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) 

 

    cv2.putText(original_frame, f"FPS: {fps:.1f}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 

    cv2.putText(original_frame, f"Latency: {latency_ms:.1f} ms", (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2) 

 

    cv2.imshow("Gesture Control Optimized", original_frame) 

 

    if cv2.waitKey(1) & 0xFF == ord("q"): 

        break 

 

camera.stop() 

cv2.destroyAllWindows() 
