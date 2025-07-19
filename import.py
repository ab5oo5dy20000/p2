import cv2
import os

# ✅ مسار الفيديو الصحيح
video_path = r"D:\python_prog\video1.mp4"  # اسم الفيديو الصحيح

# ✅ مجلد لحفظ الإطارات
output_folder = r"D:\python_prog\frames"

# إنشاء المجلد إذا ما كان موجود
os.makedirs(output_folder, exist_ok=True)

# فتح الفيديو
vidcap = cv2.VideoCapture(video_path)

# تأكد أن الفيديو انفتح بنجاح
if not vidcap.isOpened():
    print("❌ فشل في فتح الفيديو. تحقق من المسار أو اسم الملف.")
    exit()

# عداد الإطارات
frame_count = 0

# قراءة أول إطار
success, image = vidcap.read()

while success:
    # اسم الإطار مع تنسيق الأرقام: frame_00001.jpg
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, image)

    # قراءة الإطار التالي
    success, image = vidcap.read()
    frame_count += 1

# النتيجة
print(f"✅ تم استخراج {frame_count} إطار داخل: {output_folder}")
