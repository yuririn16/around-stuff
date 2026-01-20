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
    # 1. 画像の準備
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 2. 検出器の作成と実行
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=score_threshold,
    )

    # ここからAIの実行
    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            detection_result = detector.detect(mp_image)

            # 3. 描画の準備
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            font = ImageFont.load_default()

            if detection_result.detections:
                colors = ["#00FF00", "#FF4B4B", "#1C83E1", "#FFD700", "#FF00FF"]
                
                for i, detection in enumerate(detection_result.detections):
                    color = colors[i % len(colors)]
                    bbox = detection.bounding_box
                    
                    # 枠の計算
                    left, top = bbox.origin_x, bbox.origin_y
                    right, bottom = left + bbox.width, top + bbox.height
                    draw.rectangle([left, top, right, bottom], outline=color, width=5)
                    
                    # ラベルの作成
                    cat = detection.categories[0]
                    name = LABEL_MAP.get(cat.category_name, cat.category_name)
                    label = f"{name} {int(cat.score * 100)}%"
                    
                    # ラベル背景と文字
                    draw.rectangle([left, top - 25, left + len(label)*10, top], fill=color)
                    draw.text((left + 2, top - 22), label, fill="white")
                
                st.image(draw_image, use_container_width=True)
                
                # レポート表示
                st.subheader("📊 検出レポート")
                for detection in detection_result.detections:
                    cat = detection.categories[0]
                    disp_name = LABEL_MAP.get(cat.category_name, cat.category_name)
                    st.write(f"**{disp_name}**")
                    st.progress(float(cat.score))
            else:
                st.image(image, use_container_width=True)
                st.warning("検知されませんでした。")

    except Exception as e:
