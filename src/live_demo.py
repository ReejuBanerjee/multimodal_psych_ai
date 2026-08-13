# Author: Reeju Banerjee & Shourya
# Project: Explainable Multimodal AI Framework - Demo Streamer with Keyboard Override

import os
import cv2
import torch
import torchaudio
import numpy as np
from torchvision import transforms
import sounddevice as sd
from pynput import mouse, keyboard
from models import MultimodalPsychNet

class LiveTelemetryTracker:
    def __init__(self):
        self.key_press_count = 0
        self.mouse_move_count = 0
        self.last_time = cv2.getTickCount()
        
        self.mouse_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def on_move(self, x, y):
        self.mouse_move_count += 1

    def on_click(self, x, y, button, pressed):
        self.mouse_move_count += 2

    def on_press(self, key):
        self.key_press_count += 1

    def get_features(self):
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency()
        elapsed = (current_time - self.last_time) / fps
        
        typing_freq = self.key_press_count / max(elapsed, 0.1)
        mouse_freq = self.mouse_move_count / max(elapsed, 0.1)
        
        self.key_press_count = 0
        self.mouse_move_count = 0
        self.last_time = current_time
        
        features = [typing_freq, mouse_freq] + [0.0] * 16
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

def run_live_stream():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    weights_path = os.path.join(project_root, 'psych_model_weights.pth')
    
    if not os.path.exists(weights_path):
        print("[ERROR] Please train the model first to generate psych_model_weights.pth!")
        return

    # 1. Load trained model onto GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPsychNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Vision Transform pipeline
    vision_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 3. Audio transform pipeline (MFCC)
    audio_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, 
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
    )

    telemetry = LiveTelemetryTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("\n[INFO] Starting Live Stream with Presentation Override...")
    print("[CONTROLS] Press '0': Healthy | '1': Mild | '2': Moderate | '3': Severe | 'r': Reset to AI Auto | 'q': Quit\n")

    idx_to_label = {0: 'Healthy', 1: 'Mild Stress', 2: 'Moderate Stress', 3: 'Severe Stress'}
    label_colors = {
        0: (0, 255, 0),     # Green
        1: (0, 255, 255),   # Yellow
        2: (0, 165, 255),   # Orange
        3: (0, 0, 255)      # Red
    }

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    audio_sample_rate = 16000
    audio_duration = 0.5  
    temperature = 5.0

    # Override mode variable (-1 means normal AI inference)
    forced_class = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # Capture live audio chunk
        try:
            audio_chunk = sd.rec(int(audio_sample_rate * audio_duration), samplerate=audio_sample_rate, channels=1, dtype='float32')
            sd.wait()
            waveform = torch.tensor(audio_chunk).T
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            audio_feat = audio_transform(waveform)
            a_tensor = torch.mean(audio_feat, dim=2).squeeze().unsqueeze(0).to(device)
            if a_tensor.ndim == 1:
                a_tensor = a_tensor.unsqueeze(0)
        except Exception:
            a_tensor = torch.zeros(1, 40).to(device)

        t_tensor = telemetry.get_features().to(device)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.convertScaleAbs(face_roi, alpha=1.5, beta=10)
            v_tensor = vision_transform(face_roi).unsqueeze(0).to(device)

            with torch.no_grad():
                class_preds, reg_preds = model(v_tensor, a_tensor, t_tensor)
                
                scaled_preds = class_preds / temperature
                probabilities = torch.softmax(scaled_preds, dim=1)
                
                # Check if manual override is active
                if forced_class != -1:
                    pred_idx = forced_class
                    conf = 98.5 # Simulated confident presentation score
                else:
                    pred_idx = torch.argmax(probabilities, dim=1).item()
                    conf = probabilities[0][pred_idx].item() * 100
                    
                scores = reg_preds.squeeze().cpu().numpy()

            status_text = f"{idx_to_label[pred_idx]} ({conf:.1f}%)"
            box_color = label_colors.get(pred_idx, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            score_summary = f"Dep: {scores[0]:.1f} | Anx: {scores[1]:.1f} | Str: {scores[2]:.1f}"
            cv2.putText(frame, score_summary, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dashboard header and mode tracker
        mode_str = "Mode: Manual Override" if forced_class != -1 else "Mode: Live AI Stream"
        cv2.putText(frame, f"Multimodal Psych AI ({mode_str})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Live Multimodal Stress Monitor', frame)

        # Handle keyboard key inputs for presentation controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('0'):
            forced_class = 0
            print("[DEMO] Forced state: Healthy")
        elif key == ord('1'):
            forced_class = 1
            print("[DEMO] Forced state: Mild Stress")
        elif key == ord('2'):
            forced_class = 2
            print("[DEMO] Forced state: Moderate Stress")
        elif key == ord('3'):
            forced_class = 3
            print("[DEMO] Forced state: Severe Stress")
        elif key == ord('r'):
            forced_class = -1
            print("[DEMO] Reset to Auto AI Mode")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_stream()