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
st.title("🚀 AI物体検出カメラ")

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

    # AI設定
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=0.3
    )

    # 3. 実行
    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            res = detector.detect(mp_image)
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)
            
            if res.detections:
                for i, det in enumerate(res.detections):
                    # 枠の座標
                    box = det.bounding_box
                    x = box.origin_x
                    y = box.origin_y
                    w = box.width
                    h = box.height
                    
                    # 枠を描画
                    draw.rectangle([x, y, x + w, y + h], outline="#00FF00", width=5)
                    
                    # ラベル作成
                    cat = det.categories[0]
                    name = LABEL_MAP.get(cat.category_name, cat.category_name)
                    score = int(cat.score * 100)
                    txt = f"{name} {score}%"
                    
                    # ラベルの背景（読みやすくするため）
                    # 座標を計算してから描画
                    bg_x1 = x
                    bg_y1 = y - 30
                    bg_x2 = x + (len(txt) * 16)
                    bg_y2 = y
                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="#00FF00")
                    
                    # 文字を描画
                    draw.text((x + 5, y - 25), txt, fill="white")
                
                st.image(draw_img, use_container_width=True)
                
                # レポート
                st.subheader("📊 検出レポート")
                for det in res.detections:
                    c = det.categories[0]
                    n = LABEL_MAP.get(c.category_name, c.category_name)
                    st.write(f"**{n}** ({int(c.score*100)}%)")
                    st.progress(float(c.score))
            else:
                st.image(image, use_container_width=True)
                st.warning("何も見つかりませんでした。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
