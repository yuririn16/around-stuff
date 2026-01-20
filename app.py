import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import urllib.request
import cv2  # 枠を描くために使用

st.title("物体検出カメラ（枠線表示版）")

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
    # OpenCVで加工するためにコピーを作成
    image_np = np.array(image)
    # 表示用（枠を描き込む用）の画像
    output_image = image_np.copy()
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. オブジェクト検出の設定
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=0.5,
    )

    # 4. 検出実行
    with vision.ObjectDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

        # 5. 結果の描画
        if detection_result.detections:
            for detection in detection_result.detections:
                # --- A. 座標の取得 ---
                bbox = detection.bounding_box
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

                # --- B. 枠を描く (緑色の線、太さ3) ---
                cv2.rectangle(output_image, start_point, end_point, (0, 255, 0), 3)

                # --- C. 名前とスコアの取得 ---
                category = detection.categories[0]
                label_text = f"{category.category_name}: {int(category.score*100)}%"

                # --- D. ラベルを画像に書き込む ---
                cv2.putText(output_image, label_text, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 最終的な画像
