import cv2
import mediapipe as mp
import numpy as np
import os

# إعداد Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# استخراج شكل اليد كنقاط مُطبّعة (Normalized hand shape)
def extract_hand_landmarks_from_result(hand_landmarks):
    points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    center = np.mean(points, axis=0)
    norm_points = points - center
    norm_points /= np.linalg.norm(norm_points) + 1e-6
    return norm_points.flatten()

# تحميل البيانات المرجعية من صور اليد
def load_reference_data(folder_path):
    reference = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    norm_landmarks = extract_hand_landmarks_from_result(hand_landmarks)
                    reference.append(norm_landmarks)
    return reference

# دالة مقارنة اليد الحالية مع كل إشارة مرجعية
def is_hello_gesture(current_norm, reference_data, threshold=0.65):
    for ref_norm in reference_data:
        if len(current_norm) != len(ref_norm):
            continue
        diff = np.linalg.norm(current_norm - ref_norm)
        similarity = 1 - diff
        if similarity >= threshold:
            return True
    return False

# تحميل الإشارات المرجعية
reference_dir = r"D:\python_prog\dataset\hello"
reference_landmarks = load_reference_data(reference_dir)
print(f"✅ تم تحميل {len(reference_landmarks)} وضعية يد من مجلد hello")

# الكاميرا
cap = cv2.VideoCapture(0)
print("📷 الكاميرا تعمل الآن... اضغط q للخروج")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    text = "I DON'T UNDERSTAND"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            norm_landmarks = extract_hand_landmarks_from_result(hand_landmarks)
            if is_hello_gesture(norm_landmarks, reference_landmarks, threshold=0.65):
                text = "HELLO"

    # عرض النتيجة
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
    cv2.imshow("📷 التعرف على إشارة HELLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
