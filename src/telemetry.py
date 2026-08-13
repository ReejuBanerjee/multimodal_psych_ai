# Author: Reeju Banerjee & Shourya
# Project: Live Telemetry & Behavioral Tracker

import time
from pynput import mouse, listener

class LiveTelemetryTracker:
    def __init__(self):
        self.key_press_count = 0
        self.mouse_move_count = 0
        self.last_time = time.time()
        
        # Start background listeners
        self.keyboard_listener = mouse.Listener(on_move=self.on_move)
        # Note: In production, integrate pynput.keyboard for typing speed
        
    def on_move(self, x, y):
        self.mouse_move_count += 1

    def get_live_features(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        # Calculate mock/real frequencies to build an 18-dim vector matching your training schema
        typing_speed = self.key_press_count / max(elapsed, 1)
        mouse_activity = self.mouse_move_count / max(elapsed, 1)
        
        # Reset counters
        self.key_press_count = 0
        self.mouse_move_count = 0
        self.last_time = current_time
        
        # Return a tensor of shape [18] matching your training features
        features = [typing_speed, mouse_activity] + [0.0] * 16
        return features