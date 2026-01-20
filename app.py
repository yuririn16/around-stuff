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
st.title("🚀 日本語対応・AI物体検出カメラ")

# --- サイドバー設定 ---
st.sidebar.header("🛠 アプリ設定")
score_threshold = st.sidebar.slider("検知の厳しさ", 0.0, 1.0, 0.4, 0.05)

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
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    
    # 描画用の準備 (Pillowを使用)
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # フォントの設定（Streamlit Cloud環境にある標準フォントを指定）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 25)
    except:
        font = ImageFont.load_default()

    # MediaPipe用の画像変換
    image_np = np.array(image).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=score_threshold,
    )

    with vision.ObjectDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            colors = ["#00FF00", "#FF4B4B", "#1C83E1", "#FFD700", "#FF00FF"]
            
            for i, detection in enumerate(detection_result.detections):
                color = colors[i % len(colors)]
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                
                #
