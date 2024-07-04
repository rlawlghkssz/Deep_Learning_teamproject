import os
import json
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image

def convert_to_yolo_format(json_data, img_width, img_height):
    yolo_data = []
    items = json_data['FILE'][0]['ITEMS']
    for item in items:
        class_name = item['PACKAGE']
        bbox = item['BOX']
        
        # 바운딩 박스 정보를 파싱
        xmin, ymin, width, height = map(float, bbox.split(','))
        xmax = xmin + width
        ymax = ymin + height

        # YOLO 형식으로 변환
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width /= img_width
        height /= img_height

        # 클래스 이름을 숫자로 매핑
        class_id = 0 if class_name == "정상차량" else 1
        
        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_data

def save_yolo_label(yolo_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in yolo_data:
            f.write(f"{line}\n")

# 데이터 경로 설정
data_path = 'data'
output_path = 'yolo_data'

# 폴더 생성
os.makedirs(output_path, exist_ok=True)

for class_folder in ['정상차량', '불법차량']:
    img_folder = os.path.join(data_path, class_folder)
    for file_name in os.listdir(img_folder):
        if file_name.endswith('.json'):
            json_file = os.path.join(img_folder, file_name)
            img_file = os.path.join(img_folder, file_name.replace('.json', '.jpg'))

            # 이미지 크기 얻기
            img = Image.open(img_file)
            img_width, img_height = img.size

            # JSON 파일 읽기
            with open(json_file, encoding='utf-8') as f:
                json_data = json.load(f)

            yolo_data = convert_to_yolo_format(json_data, img_width, img_height)
            output_file = os.path.join(output_path, file_name.replace('.json', '.txt'))
            save_yolo_label(yolo_data, output_file)

            # 이미지 복사
            copyfile(img_file, os.path.join(output_path, os.path.basename(img_file)))

# 데이터셋 분리
# 이미지와 라벨 리스트
images = [f for f in os.listdir(output_path) if f.endswith('.jpg')]
labels = [f for f in os.listdir(output_path) if f.endswith('.txt')]

# 데이터셋 분리
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

def move_files(file_list, source_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for file_name in file_list:
        copyfile(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))
        copyfile(os.path.join(source_folder, file_name.replace('.jpg', '.txt')), os.path.join(dest_folder, file_name.replace('.jpg', '.txt')))

# 파일 이동
move_files(train_images, output_path, 'yolo_data/train')
move_files(test_images, output_path, 'yolo_data/test')
