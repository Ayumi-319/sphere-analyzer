import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

st.set_page_config(page_title="Sphere Analyzer v4.9", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.9 (切り離し強化版)")

if 'bg_colors' not in st.session_state:
    st.session_state.bg_colors = []

with st.sidebar:
    st.header("1. 基本設定 (確定スケール)")
    mag = st.radio("倍率:", ("4x", "10x", "20x", "カスタム"), index=1) # デフォルト10x
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 1.25)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 4.21)

    st.header("3. 二値化 (スポイト)")
    if st.button("スポイト履歴をリセット"):
        st.session_state.bg_colors = []
    
    sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 4) # スクショの設定反映
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 0.0)

    st.header("4. ノイズ処理")
    remove_white = st.slider("⚪ 白ゴミ取り", 0, 1000, 380) # スクショ反映
    remove_black = st.slider("⚫ 黒穴埋め", 0, 2000, 195) # 最大値を広げました

    st.header("5. 強力切り離し (雪だるま対策)")
    # Watershedの感度を調整する新スライダー
    watershed_footprint = st.slider("切り離し感度 (Peak Footprint)", 1, 100, 30, help="数値を下げると、より積極的に雪だるまを分割します")
    min_dist = st.slider("中心間の最小距離 (px)", 1, 200, 30)

    st.header("6. フィルタ")
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    original_w, _ = img_raw.size
    preview_img = img_raw.copy()
    preview_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    scale_factor = original_w / preview_img.size[0]
    
    img_edit = preview_img.copy()
    if invert_image: img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    
    col_img, col_res = st.columns([2, 1])
    
    with col_img:
        st.subheader("背景を数カ所クリックしてください")
        coords = im_coordinates(img_edit, key="coords")
        if coords:
            temp_gray = color.rgb2gray(np.array(img_edit))
            st.session_state.bg_colors.append(temp_gray[int(coords['y']), int(coords['x'])])

    gray_img = color.rgb2gray(np.array(img_edit))
    if blur_sigma > 0: gray_img = filters.gaussian(gray_img, sigma=blur_sigma)

    if st.session_state.bg_colors:
        diff_map = np.ones_like(gray_img)
        for bg_c in st.session_state.bg_colors:
            diff_map = np.minimum(diff_map, np.abs(gray_img - bg_c))
        binary = diff_map > (sensitivity / 255.0)
        
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white)
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black)
        
        # --- 強化された切り離しロジック ---
        distance = ndi.distance_transform_edt(filled)
        # Peak discovery を強化
        local_maxi = feature.peak_local_max(distance, min_distance=min_dist, 
                                            footprint=np.ones((watershed_footprint, watershed_footprint)), 
                                            labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        # ------------------------------

        props = measure.regionprops(labels)
        df_list = []
        for p in props:
            if p.area >= min_area:
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
                if circ > circularity_threshold:
                    df_list.append({'label': p.label, 'diam': p.equivalent_diameter * um_per_pixel * scale_factor})
        df_base = pd.DataFrame(df_list)

        with col_res:
            st.header("📊 結果")
            if not df_base.empty:
                st.metric("検出数", f"{len(df_base)} 個")
                st.metric("平均直径", f"{df_base['diam'].mean():.1f} μm")
                fig, ax = plt.subplots()
                ax.imshow(np.array(img_edit))
                ax.contour(labels > 0, colors='lime', linewidths=1.0)
                ax.axis('off')
                st.pyplot(fig)
