# Author: Reeju Banerjee & Shourya
# Project: Explainable Multimodal AI Framework - Modern GUI Dashboard

import os
import cv2
import torch
import torchaudio
import numpy as np
import sounddevice as sd
from pynput import mouse, keyboard
from torchvision import transforms
from PIL import Image
import customtkinter as ctk

# Matplotlib configuration for GUI integration
import matplotlib
matplotlib.use('Agg') # Forces matplotlib to render in the background without opening a clashing window
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet

# Set modern dark theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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


class MultimodalGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Multimodal Psych AI - Clinical Dashboard")
        self.geometry("1100x750")
        
        # --- UI LAYOUT ---
        # Left Panel for Webcam
        self.video_frame = ctk.CTkFrame(self, width=680, height=520)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing Camera...")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Right Panel for Telemetry & Diagnostics
        self.stats_frame = ctk.CTkFrame(self, width=350)
        self.stats_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.stats_frame, text="Diagnostic Output", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)
        
        self.status_label = ctk.CTkLabel(self.stats_frame, text="Status: Calibrating...", font=ctk.CTkFont(size=20), text_color="#deff9a")
        self.status_label.pack(pady=10)
        
        self.conf_label = ctk.CTkLabel(self.stats_frame, text="Confidence: --%", font=ctk.CTkFont(size=16))
        self.conf_label.pack(pady=10)
        
        # Regression Scores
        self.dep_label = ctk.CTkLabel(self.stats_frame, text="Depression Index: --", font=ctk.CTkFont(size=16))
        self.dep_label.pack(pady=5)
        self.anx_label = ctk.CTkLabel(self.stats_frame, text="Anxiety Index: --", font=ctk.CTkFont(size=16))
        self.anx_label.pack(pady=5)
        self.str_label = ctk.CTkLabel(self.stats_frame, text="Stress Index: --", font=ctk.CTkFont(size=16))
        self.str_label.pack(pady=5)
        
        # --- OVERRIDE CONTROLS ---
        self.override_label = ctk.CTkLabel(self.stats_frame, text="Manual Override Controls", font=ctk.CTkFont(size=14, weight="bold"))
        self.override_label.pack(pady=(20, 5))
        
        self.btn_0 = ctk.CTkButton(self.stats_frame, text="Force: Healthy", fg_color="#2E8B57", hover_color="#3CB371", command=lambda: self.set_override(0))
        self.btn_0.pack(pady=5)
        
        self.btn_1 = ctk.CTkButton(self.stats_frame, text="Force: Mild Stress", fg_color="#B8860B", hover_color="#DAA520", command=lambda: self.set_override(1))
        self.btn_1.pack(pady=5)
        
        self.btn_2 = ctk.CTkButton(self.stats_frame, text="Force: Mod. Stress", fg_color="#D2691E", hover_color="#CD853F", command=lambda: self.set_override(2))
        self.btn_2.pack(pady=5)
        
        self.btn_3 = ctk.CTkButton(self.stats_frame, text="Force: Severe Stress", fg_color="#8B0000", hover_color="#A52A2A", command=lambda: self.set_override(3))
        self.btn_3.pack(pady=5)
        
        self.reset_btn = ctk.CTkButton(self.stats_frame, text="Reset to Live AI Mode", fg_color="#333333", command=self.reset_mode)
        self.reset_btn.pack(pady=15)
        
        # --- BATCH EVALUATION BUTTON ---
        self.grid_btn = ctk.CTkButton(self.stats_frame, text="Generate Batch Inference Grid", fg_color="#4B0082", hover_color="#8A2BE2", command=self.show_inference_grid)
        self.grid_btn.pack(side="bottom", pady=20)
        
        # --- AI INITIALIZATION ---
        self.init_ai()
        self.forced_class = -1
        
        # Start Video Loop
        self.update_frame()

    def init_ai(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        weights_path = os.path.join(self.project_root, 'psych_model_weights.pth')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultimodalPsychNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.vision_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

        # Load dataset for the Batch Grid
        test_csv = os.path.join(self.project_root, 'data', 'tabular', 'test_data.csv')
        audio_dir = os.path.join(self.project_root, 'data', 'audio')
        vision_dir = os.path.join(self.project_root, 'data', 'vision')
        
        self.test_dataset = PsychMultimodalDataset(test_csv, audio_dir, vision_dir)

        self.telemetry = LiveTelemetryTracker()
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.idx_to_label = {0: 'Healthy', 1: 'Mild Stress', 2: 'Moderate Stress', 3: 'Severe Stress'}
        self.colors = {0: "#00FF00", 1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}

    def reset_mode(self):
        self.forced_class = -1

    def set_override(self, class_idx):
        self.forced_class = class_idx

    def show_inference_grid(self):
        # 1. Grab random batch
        test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=True)
        images, audios, tabulars, true_classes, _ = next(iter(test_loader))
        
        images_dev = images.to(self.device)
        audios_dev = audios.to(self.device)
        tabulars_dev = tabulars.to(self.device)
        
        with torch.no_grad():
            class_preds, _ = self.model(images_dev, audios_dev, tabulars_dev)
            probabilities = torch.softmax(class_preds, dim=1)
            confs, pred_classes = torch.max(probabilities, 1)
            
        # 2. Render plot in background
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.patch.set_facecolor('#2b2b2b')
        fig.suptitle('Test Set Inference Grid', fontsize=16, fontweight='bold', color='white', y=0.95)
        axes = axes.flatten()
        
        for i in range(8):
            ax = axes[i]
            img = images[i].squeeze().numpy()
            img = (img * 0.5) + 0.5
            ax.imshow(img, cmap='gray')
            
            true_label = self.idx_to_label[true_classes[i].item()]
            pred_label = self.idx_to_label[pred_classes[i].item()]
            conf_score = confs[i].item() * 100
            
            color = '#00FF00' if true_label == pred_label else '#FF0000'
            title = f"True: {true_label}\nPred: {pred_label}\n({conf_score:.1f}%)"
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
            
        plt.tight_layout()
        grid_path = os.path.join(self.project_root, 'temp_gui_grid.png')
        plt.savefig(grid_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        
        # 3. Create Pop-up Window
        popup = ctk.CTkToplevel(self)
        popup.title("Batch Inference Visualizer")
        popup.geometry("900x500")
        popup.attributes('-topmost', True) # Keeps it in front
        
        # 4. Load and display image
        grid_img = Image.open(grid_path)
        ctk_grid = ctk.CTkImage(light_image=grid_img, dark_image=grid_img, size=(850, 450))
        img_label = ctk.CTkLabel(popup, image=ctk_grid, text="")
        img_label.pack(padx=20, pady=20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(48, 48))

            # Audio processing
            try:
                audio_chunk = sd.rec(int(16000 * 0.5), samplerate=16000, channels=1, dtype='float32')
                sd.wait()
                waveform = torch.tensor(audio_chunk).T
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                audio_feat = self.audio_transform(waveform)
                a_tensor = torch.mean(audio_feat, dim=2).squeeze().unsqueeze(0).to(self.device)
                if a_tensor.ndim == 1: a_tensor = a_tensor.unsqueeze(0)
            except:
                a_tensor = torch.zeros(1, 40).to(self.device)

            t_tensor = self.telemetry.get_features().to(self.device)

            for (x, y, w, h) in faces:
                face_roi = cv2.convertScaleAbs(gray_frame[y:y+h, x:x+w], alpha=1.5, beta=10)
                v_tensor = self.vision_transform(face_roi).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    class_preds, reg_preds = self.model(v_tensor, a_tensor, t_tensor)
                    probabilities = torch.softmax(class_preds / 5.0, dim=1)
                    
                    if self.forced_class != -1:
                        pred_idx = self.forced_class
                        conf = 99.9
                    else:
                        pred_idx = torch.argmax(probabilities, dim=1).item()
                        conf = probabilities[0][pred_idx].item() * 100
                        
                    scores = reg_preds.squeeze().cpu().numpy()

                label_text = self.idx_to_label[pred_idx]
                hex_color = self.colors.get(pred_idx, "#FFFFFF")
                
                # Corrected OpenCV Color Bounding Box
                rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                bgr_color = rgb_color[::-1] 
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, 2)
                
                self.status_label.configure(text=f"Status: {label_text}", text_color=hex_color)
                self.conf_label.configure(text=f"Confidence: {conf:.1f}%")
                self.dep_label.configure(text=f"Depression Index: {scores[0]:.2f}")
                self.anx_label.configure(text=f"Anxiety Index: {scores[1]:.2f}")
                self.str_label.configure(text=f"Stress Index: {scores[2]:.2f}")

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            
            self.video_label.configure(image=ctk_image, text="")

        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = MultimodalGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()