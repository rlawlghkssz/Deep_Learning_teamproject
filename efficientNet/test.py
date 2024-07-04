import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
data_dir = 'data'

# 이미지와 라벨 리스트 생성
images = []
labels = []

# JSON 파일에서 클래스 정보 추출 함수
def extract_info_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        file_info = data["FILE"][0]
        item_info = file_info["ITEMS"][0]
        label = 0 if item_info["CLASS"] == "정상차량" else 1
        return label

# 데이터 폴더 탐색
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(class_path, filename)
                json_path = img_path.replace('.jpg', '.json').replace('.png', '.json')
                images.append(img_path)
                label = extract_info_from_json(json_path)
                labels.append(label)

# 데이터셋을 학습/검증 세트로 분리
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# 데이터 증강 및 제너레이터 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

def create_data_generator(datagen, images, labels, batch_size=32):
    while True:
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_images = []
            batch_labels = labels[start:end]
            for img_path in images[start:end]:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                batch_images.append(img)
            yield np.vstack(batch_images), np.array(batch_labels)

batch_size = 32
train_generator = create_data_generator(train_datagen, train_images, train_labels, batch_size)
val_generator = create_data_generator(val_datagen, val_images, val_labels, batch_size)

# EfficientNetB0 모델 불러오기 (사전 학습된 가중치 사용)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 새로운 출력층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# 전체 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 사전 학습된 층은 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_images) // batch_size
)

# 모델 평가 및 결과 출력
import matplotlib.pyplot as plt

# 모델 평가
loss, accuracy = model.evaluate(val_generator, steps=len(val_images) // batch_size)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')

# 학습 과정 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
