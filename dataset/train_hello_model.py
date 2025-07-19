import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ مسار البيانات
data_path = r"D:\python_prog\dataset"

# ⚠️ تأكد أن داخل هذا المسار مجلد اسمه "hello" فيه الصور

# إعداد الصور
img_size = (64, 64)
batch_size = 16

# تجهيز البيانات
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% تدريب، 20% اختبار
)

train_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ✅ بناء نموذج CNN بسيط
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# تجميع النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(train_generator, validation_data=val_generator, epochs=10)

# حفظ النموذج
model.save(r"D:\python_prog\hello_model.h5")

print("✅ تم تدريب النموذج وحفظه كـ hello_model.h5")
