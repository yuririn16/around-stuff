import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import urllib.request
import os

# 日本語辞書
LABEL_MAP = {
    "person": "人間", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "airplane": "飛行機", "bus": "バス", "train": "電車", "truck": "トラック",
    "bottle": "ボトル", "wine glass": "グラス", "cup": "コップ", "fork": "フォーク",
    "knife": "ナイフ", "spoon": "スプーン", "bowl": "ボウル", "banana": "バナナ",
    "apple": "りんご", "chair": "椅子", "couch": "ソファ", "potted plant": "観葉植物",
    "tv": "テレビ", "laptop": "PC", "mouse": "マウス", "remote": "リモコン",
    "keyboard": "キーボード", "cell phone": "スマホ", "book": "本", "clock": "時計"
}

st.set_page_config(page_title="AI物体検出", layout="centered")
st.title("🎨 カラー別・AI物体検出カメラ")

# 1. モデル準備
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

model_file = load_model_file()

# 2. カメラ入力
img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_
