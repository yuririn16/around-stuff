import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import cv2
import os

st.title("物体検出カメラ（リスト表示版）")

# 1. モデルファイルの準備
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        with st.spinner("AIモデルを準備中..."):
            urllib.request.urlretrieve(model_url, model_path)
    return model_path

try:
    model_file = load_model_file()
except Exception as e:
    st.error(f"モデルのダウンロードに失敗しました: {e}")

# 2. カメラ入力
img_file = st.camera_input("写真を撮る")

if img_file is not None:
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image).astype(np.uint8)
    
    output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. 検出器の設定
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.2, 
        max_results=10        # リスト化するので少し多めに表示可能に
    )

    # 4. 実行と結果表示
    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            detection_result = detector.detect(mp_image)

            if detection_result.detections:
                # 枠線の色リスト
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
                
                # 見つかった物体の名前を保存するリスト
                found_labels = []
                
                for i, detection in enumerate(detection_result.detections):
                    color = colors[i % len(colors)]
                    bbox = detection.bounding_box
                    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                    
                    # 枠とラベルを画像に描画
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
                    category = detection.categories[0]
