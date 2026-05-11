import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps

st.set_page_config(page_title="Sphere Analyzer v4.1", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.1 (真円フィッティング版)")

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=False)
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.0)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0)

    st.header("3. 二値化 (くり抜きモード)")
    extraction_mode = st.radio("抽出方法:", ("背景を一気にくり抜く (おすすめ)", "細かい影を拾う (従来)"), index=0)
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 0.0)
    
    if extraction_mode == "背景を一気にくり抜く (おすすめ)":
        global_offset = st.slider("背景の判定ライン (閾値微調整)", -0.30, 0.30, -0.13, step=0.01)
    else:
        block_size_slider = st.slider("二値化の細かさ (Block Size)", 11, 201, 51, step=10)
        local_offset = st.slider("縁の拾いやすさ (Offset)", -0.10, 0.10, 0.00, step=0.01)

    st.header("4. 高精度ノイズ処理")
    st.write("💡 スライダー右端の「数字」をクリックで手入力")
    remove_white = st.slider("⚪ 白いゴミ取り (背景ノイズ)", 0, 1000, 0, step=10)
    remove_black = st.slider("⚫ 黒いゴミ取り (内部の穴埋め)", 0, 200, 20, step=1)

    st.header("5. 切り離し (Watershed)")
    min_dist = st.number_input("中心間の最小距離 (px)", value=30)

    st.header("6. 最終足切りフィルタ")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        img_edit = img_raw.copy()
        if invert_image:
            img_edit = ImageOps.invert(img_edit)
            
        enhancer_c = ImageEnhance.Contrast(img_edit)
        img_edit = enhancer_c.enhance(contrast)
        enhancer_s = ImageEnhance.Sharpness(img_edit)
        img_edit = enhancer_s.enhance(sharpness)
        
        img_np = np.array(img_edit)
        gray = color.rgb2gray(img_np)
        
        if blur_sigma > 0:
            blurred = filters.gaussian(gray, sigma=blur_sigma)
        else:
            blurred = gray

        if extraction_mode == "背景を一気にくり抜く (おすすめ)":
            base_thresh = filters.threshold_otsu(blurred)
            adj_thresh = base_thresh + global_offset
            if invert_image:
                binary = blurred > adj_thresh
            else:
                binary = blurred < adj_thresh
        else:
            block_size = int(block_size_slider)
            if block_size % 2 == 0:
                block_size += 1
            local_thresh = filters.threshold_local(blurred, block_size, offset=local_offset)
            if invert_image:
                binary = blurred > local_thresh
            else:
                binary = blurred < local_thresh 
            
        if remove_white > 0:
            cleaned = morphology.remove_small_objects(binary, min_size=remove_white)
        else:
            cleaned = binary

        if remove_black > 0:
            filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black)
        else:
            filled = cleaned
        
        distance = ndi.distance_transform_edt(filled)
        coords = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
        
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        
        if exclude_border:
            labels = segmentation.clear_border(labels)

        st.subheader(f"解析: {f.name}")
        
        with st.expander("🔍 途中経過（補正・二値化・ゴミ取り・穴埋め）を見る"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.image(img_edit, caption="0. 補正後の画像", use_container_width=True)
            with col_b:
                st.image(binary.astype(float), caption="1. 二値化直後", use_container_width=True)
            with col_c:
                st.image(filled.astype(float), caption="2. 白/黒ゴミ取り後", use_container_width=True)

        props = measure.regionprops(labels)
        df_list = []
        
        original_w, _ = Image.open(f).size
        current_w, _ = img_raw.size
        scale_factor = original_w / current_w

        for p in props:
            area = p.area
            perimeter = p.perimeter
            if perimeter > 0 and area >= min_area:
                circ = (4 * np.pi * area) / (perimeter ** 2)
                if circ > circularity_threshold:
                    diam_um = p.equivalent_diameter * um_per_pixel * scale_factor
                    df_list.append({
                        'label': p.label,
                        'area': area,
                        'perimeter': perimeter,
                        'equivalent_diameter': p.equivalent_diameter,
                        'circularity': circ,
                        'diameter_um': diam_um,
                        'centroid_y': p.centroid[0],
                        'centroid_x': p.centroid[1]
                    })

        df_clean = pd.DataFrame(df_list)
        
        col_res
