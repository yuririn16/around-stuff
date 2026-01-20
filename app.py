import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import cv2

st.title("高視認性・物体検出カメラ")

# 1. AIモデルの準備
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
model_path = "model.tflite"

@st.cache_resource
def load_model():
    urllib.request.urlretrieve(model_url, model_path)
    return model_path

model_file = load_model()

# 2. カメラ入力
img_file = st.camera_input("写真を撮る")

if img_file is not None:
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image)
    
    # 加工用のBGR画像を作成
    output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=0.3,
    )

    with vision.ObjectDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            # 物体ごとに色を変えるための色のリスト
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

            for i, detection in enumerate(detection_result.detections):
                color = colors[i % len(colors)] # リストから色を順番に選択
                
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                
                # --- A. 枠を描く ---
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 4)

                # --- B. ラベルのテキスト作成 ---
                category = detection.categories[0]
                label_text = f"{category.category_name} {int(category.score*100)}%"
                
                # --- C. 文字の背景ボックスを描く（視認性アップの鍵） ---
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 文字が画面の上からはみ出さないように調整
                label_y = y - 10
