import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
import cv2

st.set_page_config(page_title="Sphere Analyzer v3.4", layout="wide")
st.title("🔴 スフェア自動計測ツール v3.4 (背景くり抜きモード)")

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=False)
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.0)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0)

    st.header("3. 二値化 (くり抜きモード)")
    # 今回の目玉：抽出方法の選択
    extraction_mode = st.radio("抽出方法:", ("背景を一気にくり抜く (のっぺり抽出) [ユーザー発案]", "細かい影を拾う (従来)"), index=0)
    
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 3.0, help="背景を滑らかにして分離しやすくします")
    
    if extraction_mode == "背景を一気にくり抜く (のっぺり抽出) [ユーザー発案]":
        global_offset = st.slider("背景の判定ライン (閾値微調整)", -0.30, 0.30, 0.00, step=0.01, help="右に動かすと背景が削れスフェアが太くなり、左に動かすと背景が広がります")
    else:
        block_size_slider = st.slider("二値化の細かさ (Block Size)", 11, 201, 51, step=10)
        local_offset = st.slider("縁の拾いやすさ (Offset)", -0.10, 0.10, 0.00, step=0.01)

    st.header("4. ノイズ処理 (ImageJ機能)")
    noise_removal = st.slider("ゴミ取り (最小ピクセル数)", 0, 1000, 50, help="この数値以下の小さな点を消去します")
    fill_holes = st.checkbox("スフェア内部の穴埋めを実行", value=True, help="ザラザラを潰して一つの塊にします")

    st.header("5. 切り離し (Watershed)")
    min_dist = st.number_input("中心間の最小距離 (px)", value=30)

    st.header("6. 最終足切りフィルタ")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.6)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        img_edit = img_raw.copy()
