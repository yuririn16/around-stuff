import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import urllib.request
import os

# 日本語ラベル辞書
LABEL_MAP = {
    "person": "人間", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "bottle": "ボトル", "cup": "コップ", "chair": "椅子", "tv": "テレビ",
    "laptop": "PC", "mouse": "マウス", "keyboard": "キーボード", "cell phone": "スマホ"
}

st.set_page_config(page_title="AIカメラ", layout="centered")
st.title("🎨 カラー別・AI物体検出")

# 1. モデル準備
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

try:
    m_file = load_model_file()
except:
    st.error("モデルの読み込みに失敗しました")

# 2. カメラ入力
img_file = st.camera_input("撮影する")

if img_file is not None:
    # 画像準備
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img)
    img_np = np.array(img).astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    # 3. AIの設定 (行を短く分割してエラー防止)
    base_ops = python.BaseOptions(model_asset_path=m_file)
    options = vision.ObjectDetectorOptions(
        base_options
