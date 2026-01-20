import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import urllib.request
import os

# --- 設定と日本語辞書 ---
LABEL_MAP = {
    "person": "人間", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "airplane": "飛行機", "bus": "バス", "train": "電車", "truck": "トラック",
    "bottle": "ボトル", "wine glass": "グラス", "cup": "コップ", "fork": "フォーク",
    "knife": "ナイフ", "spoon": "スプーン", "bowl": "ボウル", "banana": "バナナ",
    "apple": "りんご", "chair": "椅子", "couch": "ソファ", "potted plant": "観葉植物",
    "tv": "テレビ", "laptop": "PC", "mouse": "マウス", "remote": "リモコン",
    "keyboard": "キーボード", "cell phone": "スマホ", "book": "本", "clock": "時計"
}

st.set_page_config(page_title="AI物体検出カメラ", layout="centered")
st.title("🚀 安定版・AI物体検出カメラ")

# --- サイドバー設定 ---
st.sidebar.header("🛠 アプリ設定")
score_threshold = st.sidebar.slider("検知の厳しさ", 0.1, 1.0, 0.3, 0.05)

# --- モデル準備 ---
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

model_file = load_model_file()

# --- メイン処理 ---
img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    # 1. 画像の読み込みと向きの補正
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    
    # 2. AI用の画像データ(numpy)を準備
    # ここを確実に uint8 型にすることで検知漏れを防ぎます
    image_np = np.array(image).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. オブジェクト検出の設定
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=score_threshold,
    )

    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            detection_result = detector.detect(mp_image)

            # 4. 描画処理 (Pillowを使用)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # フォント読み込み (日本語が出ない場合はデフォルト)
            font = ImageFont.load_default()

            if detection_result.detections:
                colors = ["#00FF00", "#FF4B4B", "#1C83E1", "#FFD700", "#FF00FF"]
                
                for i, detection in enumerate(detection_result.detections):
                    color = colors[i % len(colors)]
                    bbox = detection.bounding_box
