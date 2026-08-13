# Author: Reeju Banerjee
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework

import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

def get_audio_inventories(audio_dir):
    """Dynamically splits available audio files 80/20 to prevent empty sequences."""
    train_inv = {'Healthy': [], 'Mild_Stress': [], 'Moderate_Stress': [], 'Severe_Stress': []}
    test_inv = {'Healthy': [], 'Mild_Stress': [], 'Moderate_Stress': [], 'Severe_Stress': []}
    
    emotion_to_stress = {
        '01': 'Healthy', '02': 'Healthy', '03': 'Healthy',
        '04': 'Mild_Stress', '08': 'Mild_Stress',
        '05': 'Moderate_Stress', '06': 'Moderate_Stress',
        '07': 'Severe_Stress'
    }
    
    # Temporary holding dictionary
    all_audio = {'Healthy': [], 'Mild_Stress': [], 'Moderate_Stress': [], 'Severe_Stress': []}
    seen_files = set()
    
    for root, dirs, files in os.walk(audio_dir):
        # Ignore the duplicate master folder
        if "audio_speech_actors" in root:
            continue
            
        for filename in files:
            if filename.endswith('.wav') and filename not in seen_files:
                parts = filename.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_to_stress:
                        stress_level = emotion_to_stress[emotion_code]
                        rel_path = os.path.relpath(os.path.join(root, filename), audio_dir)
                        all_audio[stress_level].append(rel_path)
                        seen_files.add(filename)
                        
    # Dynamically split 80/20 for each category
    for stress_level, files in all_audio.items():
        if not files:
            continue
            
        random.seed(42)
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.8)
        
        # Fallback to ensure test set gets at least some files if dataset is tiny
        if split_idx == len(files) and len(files) > 1:
            split_idx = len(files) - 1
            
        train_inv[stress_level].extend(files[:split_idx])
        test_inv[stress_level].extend(files[split_idx:])
        
    return train_inv, test_inv

def get_vision_inventories(vision_dir):
    """Scans vision subfolders and dynamically splits images 80/20."""
    train_inv = {'Healthy': [], 'Mild_Stress': [], 'Moderate_Stress': [], 'Severe_Stress': []}
    test_inv = {'Healthy': [], 'Mild_Stress': [], 'Moderate_Stress': [], 'Severe_Stress': []}
    
    folder_to_stress = {
        'happy': 'Healthy', 'neutral': 'Healthy',
        'sad': 'Mild_Stress', 'surprise': 'Mild_Stress',
        'fear': 'Moderate_Stress', 'disgust': 'Moderate_Stress',
        'angry': 'Severe_Stress'
    }
    
    existing_folders = {}
    if os.path.exists(vision_dir):
        for sub in os.listdir(vision_dir):
            full_sub = os.path.join(vision_dir, sub)
            if os.path.isdir(full_sub):
                existing_folders[sub.lower()] = sub
    
    for target_folder, stress_level in folder_to_stress.items():
        if target_folder in existing_folders:
            actual_folder = existing_folders[target_folder]
            folder_path = os.path.join(vision_dir, actual_folder)
            
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                continue
                
            random.seed(42)
            random.shuffle(images)
            
            split_idx = int(len(images) * 0.8)
            train_inv[stress_level].extend([f"{actual_folder}/{img}" for img in images[:split_idx]])
            test_inv[stress_level].extend([f"{actual_folder}/{img}" for img in images[split_idx:]])
                
    return train_inv, test_inv

def verify_inventory(inv_name, inventory):
    """Verifies all categories have at least 1 file."""
    is_valid = True
    print(f"\n--- {inv_name} Inventory Counts ---")
    for category, files in inventory.items():
        count = len(files)
        print(f"  {category}: {count} files")
        if count == 0:
            is_valid = False
            print(f"  [ERROR] {category} has 0 files in {inv_name}!")
    return is_valid

def assign_multimodal_files(df, audio_inv, vision_inv):
    assigned_audios = []
    assigned_images = []
    
    for index, row in df.iterrows():
        stress_level = row['Mental_Health_Status']
        # This will no longer crash because we verified the pools aren't empty!
        assigned_audios.append(random.choice(audio_inv[stress_level]))
        assigned_images.append(random.choice(vision_inv[stress_level]))
        
    df = df.copy()
    df['Assigned_Audio'] = assigned_audios
    df['Assigned_Image'] = assigned_images
    return df

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, '../data/audio')
    vision_dir = os.path.join(current_dir, '../data/vision')
    tabular_csv = os.path.join(current_dir, '../data/tabular/mental_health_multimodal.csv')
    
    output_train = os.path.join(current_dir, '../data/tabular/train_data.csv')
    output_test = os.path.join(current_dir, '../data/tabular/test_data.csv')

    print("Loading file inventories...")
    audio_train_inv, audio_test_inv = get_audio_inventories(audio_dir)
    vision_train_inv, vision_test_inv = get_vision_inventories(vision_dir)

    # Diagnostic Checks (The script will physically stop here if something is still 0)
    v1 = verify_inventory("Audio Train", audio_train_inv)
    v2 = verify_inventory("Audio Test", audio_test_inv)
    v3 = verify_inventory("Vision Train", vision_train_inv)
    v4 = verify_inventory("Vision Test", vision_test_inv)

    if not (v1 and v2 and v3 and v4):
        print("\n[CRITICAL ERROR] Script stopped. One or more categories contain 0 files.")
        return

    print("\nReading tabular data...")
    df = pd.read_csv(tabular_csv)
    
    label_map = {'Healthy': 0, 'Mild_Stress': 1, 'Moderate_Stress': 2, 'Severe_Stress': 3}
    df['Class_Label'] = df['Mental_Health_Status'].map(label_map)
    
    print("Splitting tabular data into 80% Train and 20% Test...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Mental_Health_Status'])
    
    print("Aligning multimodal sets...")
    train_df = assign_multimodal_files(train_df, audio_train_inv, vision_train_inv)
    test_df = assign_multimodal_files(test_df, audio_test_inv, vision_test_inv)
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"\nSuccess! Safely aligned dataset.")
    print(f"Generated: {output_train}")
    print(f"Generated: {output_test}")

if __name__ == "__main__":
    main()