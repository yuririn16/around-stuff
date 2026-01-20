import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import cv2
import os

# --- 設定と日本語辞書 ---
LABEL_MAP = {
    "person": "人間", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "airplane": "飛行機", "bus": "バス", "train": "電車", "truck": "トラック",
    "boat": "船", "traffic light": "信号機", "stop sign": "一時停止",
    "bench": "ベンチ", "bird": "鳥", "cat": "猫", "dog": "犬", "horse": "馬",
    "sheep": "羊", "cow": "牛", "elephant": "象", "bear": "クマ", "zebra": "シマウマ",
    "giraffe": "キリン", "backpack": "リュック", "umbrella": "傘", "handbag": "ハンドバッグ",
    "tie": "ネクタイ", "suitcase": "スーツケース", "frisbee": "フリスビー", "skis": "スキー板",
    "snowboard": "スノーボード", "sports ball": "ボール", "kite": "凧", "baseball bat": "バット",
    "baseball glove": "グローブ", "skateboard": "スケボー", "surfboard": "サーフボード",
    "tennis racket": "ラケット", "bottle": "ボトル", "wine glass": "グラス", "cup": "コップ",
    "fork": "フォーク", "knife": "ナイフ", "spoon": "スプーン", "bowl": "ボウル",
    "banana": "バナナ", "apple": "りんご", "sandwich": "サンドイッチ", "orange": "オレンジ",
    "broccoli": "ブロッコリー", "carrot": "にんじん", "hot dog": "ホットドッグ", "pizza": "ピザ",
    "donut": "ドーナツ", "cake": "ケーキ", "chair": "椅子", "couch": "ソファ",
    "potted plant": "観葉植物", "bed": "ベッド", "dining table": "机", "toilet": "トイレ",
    "tv": "テレビ", "laptop": "PC", "mouse": "マウス", "remote": "リモコン",
    "keyboard": "キーボード", "cell phone": "スマホ", "microwave": "電子レンジ",
    "oven": "オーブン", "toaster": "トースター", "sink": "シンク", "refrigerator": "冷蔵庫",
    "book": "本", "clock": "時計", "vase": "花瓶", "scissors": "ハサミ",
    "teddy bear": "ぬいぐるみ", "hair drier": "ドライヤー", "toothbrush": "歯ブラシ"
}

st.set_page_config(page_title="AI物体検出カメラ", layout="centered")
st.title("🚀 超高性能・AI物体検出カメラ")

# --- サイドバー設定 ---
st.sidebar.header("🛠 アプリ設定")
score_threshold = st.sidebar.slider("検知の厳しさ（しきい値）", 0.0, 1.0, 0.4, 0.05)
max_results = st.sidebar.slider("最大検知数", 1, 10, 5)

# --- モデル準備 ---
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        with st.spinner("AIモデルをダウンロード中..."):
            urllib.request.urlretrieve(model_url, model_path)
    return model_path

model_file = load_model_file()

# --- メイン処理 ---
img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image).astype(np.uint8)
    
    # 描画用
    output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=score_threshold,
        max_results=max_results
    )

    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            detection_result = detector.detect(mp_image)

            if detection_result.detections:
                colors = [(0, 255, 0), (255, 165, 0), (0, 191, 255), (255, 0, 255), (255, 255, 0)]
                
                # 結果を保存するリスト
                found_items = []

                for i, detection in enumerate(detection_result.detections):
                    color = colors[i % len(colors)]
                    bbox = detection.bounding_box
                    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                    
                    # 枠の描画
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 4)
                    
                    # ラベルの日本語化
                    category = detection.categories[0]
                    eng_name = category.category_name
                    jp_name = LABEL_MAP.get(eng_name, eng_name)
                    score_int = int(category.score * 100)
                    
                    label = f"{jp_name} ({score_int}%)"
                    found_items.append((jp_name, category.score))

                    # テキスト背景を描画
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_y = y - 10 if y - 10 > th else y + th + 10
                    cv2.rectangle(output_image, (x, text_y - th - 5), (x + tw, text_y + 5), color, -1)
                    cv2.putText(output_image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 画像表示
                final_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                st.image(final_image, use_container_width=True)

                # レポート表示
                st.subheader("📊 検出レポート")
                for name, score in found_items:
                    st.write(f"**{name}**")
                    st.progress(float(score))
            else:
                st.image(image_np, use_container_width=True)
                st.warning("何も検知されませんでした。しきい値を下げるか、もっと近づけてみてください。")
                
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
