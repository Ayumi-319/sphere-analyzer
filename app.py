import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
import io as python_io
from PIL import Image, ImageEnhance, ImageOps
import cv2

st.set_page_config(page_title="Sphere Analyzer v3.2", layout="wide")
st.title("🔴 スフェア自動計測ツール v3.2 (隠しフィルター全解放版)")

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=False)
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.0)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0)

    st.header("3. 二値化 (影を拾う力)")
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 2.0)
    # 追加：拾う細かさ（ブロックサイズ）
    block_size_slider = st.slider("二値化の細かさ (Block Size)", 11, 201, 51, step=10, help="小さいほど内部のザラザラなどの細かい影まで拾います")
    offset = st.slider("縁の拾いやすさ (Offset)", -0.10, 0.10, 0.00, step=0.01, help="マイナスにするとより多くの影を拾い、プラスにすると厳選します")

    st.header("4. ノイズ処理 (ImageJ機能)")
    # 追加：隠していたゴミ取りと穴埋めを解放
    noise_removal = st.slider("ゴミ取り (最小ピクセル数)", 0, 500, 50, help="この数値以下の小さな点（ザラザラ等）を消去します")
    fill_holes = st.checkbox("スフェア内部の穴埋めを実行", value=True, help="チェックを外すとスフェアの中のザラザラが残ったままになります")

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
        
        # 二値化ブロックサイズは必ず奇数にする必要がある
        block_size = int(block_size_slider)
        if block_size % 2 == 0:
            block_size += 1

        local_thresh = filters.threshold_local(blurred, block_size, offset=offset)
        
        if invert_image:
            binary = blurred > local_thresh
        else:
            binary = blurred < local_thresh 
            
        # 表に出した「ゴミ取り機能」
        if noise_removal > 0:
            cleaned = morphology.remove_small_objects(binary, min_size=noise_removal)
        else:
            cleaned = binary

        # 表に出した「穴埋め機能」
        if fill_holes:
            filled = ndi.binary_fill_holes(cleaned)
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
                st.image(img_edit, caption="0. 補正後の画像 (AIが見ているもの)", use_container_width=True)
            with col_b:
                st.image(binary.astype(float), caption="1. 二値化 (すべて拾った状態)", use_container_width=True)
            with col_c:
                st.image(filled.astype(float), caption="2. ゴミ取り＆穴埋め後", use_container_width=True)

        props = measure.regionprops_table(labels, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
        df = pd.DataFrame(props)
        
        col_res1, col_res2 = st.columns([2, 1])
        
        if not df.empty:
            df = df[df['area'] >= min_area]
            df = df[df['perimeter'] > 0]
            df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
            
            original_w, _ = Image.open(f).size
            current_w, _ = img_raw.size
            scale_factor = original_w / current_w
            df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel * scale_factor
            
            df_clean = df[df['circularity'] > circularity_threshold].copy()
            
            with col_res1:
                fig, ax = plt.subplots()
                # 今回は「加工後の画像（img_np）」の上に緑の線を引くので、反転などが確認できます
                ax.imshow(img_np) 
                ax.contour(labels > 0, colors='lime', linewidths=0.5)
                ax.axis('off')
                st.pyplot(fig)
                
            with col_res2:
                st.metric("検出数", f"{len(df_clean)} 個")
                if len(df_clean) > 0:
                    st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                    st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")
        else:
            st.warning("検出されませんでした。")
