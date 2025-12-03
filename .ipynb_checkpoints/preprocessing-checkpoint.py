'''
Preprocessing functions
'''

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import cv2

img_size = (64, 64)

'''
preprocess for ML model
'''
def preprocess_ml(data):
    resized = data.resize(img_size, Image.BILINEAR)

    rgb = resized.convert("RGB")
    img = np.array(rgb)

    smooth = cv2.GaussianBlur(img, (3,3), 0)

    final_img = Image.fromarray(smooth)
    return final_img



'''
Preprocess entire EuroSAT dataset
'''
def preprocess_EuroSAT(input_dir, ml_data):
    input_path = Path(input_dir)
    ml_path = Path(ml_data)
    
    
    ml_path.mkdir(exist_ok=True)
    
    
    class_names = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    for class_name in class_names:
        (ml_path / class_name).mkdir(exist_ok=True)
    
    for class_name in class_names:
        class_dir = input_path / class_name
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            
            try:
                with Image.open(img_file) as img:
                    ml_img = preprocess_ml(img)
                    ml_img.save(ml_path / class_name / img_file.name)

            except IOError as e:
                print(f"Could not read {img_file}: {e}")
                continue
            
    
    return class_names
