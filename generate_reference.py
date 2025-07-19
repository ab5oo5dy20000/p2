import cv2
import mediapipe as mp
import numpy as np
import os
import json

# إعداد Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# استخراج الإحداثيات المطبعة من يد واحدة
def extract_normalized_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            center = np.mean(points, axis=0)
            norm_points = points - center
            norm_points /= np.linalg.norm(norm_points) + 1e-6
            return norm_points.flatten().tolist()
    return None

# تحميل جميع الصور وتحويلها إلى بيانات
def generate_reference_json(folder_path, output_file):
    all_landmarks = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            data = extract_normalized_landmarks(image)
            if data:
                all_landmarks.append(data)
            else:
                print(f"⚠️ لم يتم التعرف على يد في الصورة: {filename}")

    # حفظ البيانات في JSON
    with open(output_file, 'w') as f:
        json.dump(all_landmarks, f)

    print(f"✅ تم حفظ {len(all_landmarks)} فريم في {output_file}")

# 🔧 مسار مجلد الصور + اسم ملف الإخراج
input_folder = r"D:\python_prog\dataset\hello"  # ضع هنا مسار صورك
output_json = "hello_reference.json"

generate_reference_json(input_folder, output_json)
