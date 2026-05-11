import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
import io as python_io
from PIL import Image
import cv2

st.set_page_config(page_title="Sphere Analyzer v3.0", layout="wide")
st.title("🔴 スフェア自動計測ツール v3.0 (爆速・Watershed版)")

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. 画像の二値化 (白黒にする)")
    blur_sigma = st.slider("ぼかし強さ (内部のノイズ消し)", 1.0, 10.0, 3.0, help="大きいほど中のザラザラが消えます")
    offset = st.slider("縁の拾いやすさ", -0.05, 0.05, 0.00, step=0.01, help="スフェアの輪郭がうまく線にならない時に微調整します")

    st.header("3. 切り離し設定 (Watershed)")
    min_dist = st.number_input("スフェアの中心間の最小距離 (px)", value=30, help="このピクセル数より近いものは「1つのスフェア」とみなして合体させます。細切れになる場合は数値を上げてください。")

    st.header("4. フィルタ設定 (足切り)")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100, help="これより小さいゴミを除外します")
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.6)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        # 画像読み込み (処理を軽くするため最大1000pxに制限)
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        img_np = np.array(img_raw)
        
        # --- ここからImageJと同じ処理 ---
        gray = color.rgb2gray(img_np)
        blurred = filters.gaussian(gray, sigma=blur_sigma)
        
        # 適応的閾値処理（局所的な明るさの違いに対応）
        block_size = 51
        local_thresh = filters.threshold_local(blurred, block_size, offset=offset)
        # 位相差は縁が暗いので、閾値より暗い部分をTrue（白）にする
        binary = blurred < local_thresh 
        
        # ノイズ除去と穴埋め（縁の中を塗りつぶす）
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        filled = ndi.binary_fill_holes(cleaned)
        
        # 距離変換とWatershed（切り離し）
        distance = ndi.distance_transform_edt(filled)
        coords = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
        
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        
        # 端の除外
        if exclude_border:
            labels = segmentation.clear_border(labels)
        # ---------------------------------

        st.subheader(f"解析: {f.name}")
        
        # デバッグ用（ImageJでいう途中経過の確認）
        with st.expander("🔍 途中経過（二値化・塗りつぶし）を見る"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(cleaned.astype(float), caption="1. 縁の抽出", use_container_width=True)
            with col_b:
                st.image(filled.astype(float), caption="2. 穴埋め後", use_container_width=True)

        # 計測
        props = measure.regionprops_table(labels, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
        df = pd.DataFrame(props)
        
        col_res1, col_res2 = st.columns([2, 1])
        
        if not df.empty:
            df = df[df['area'] >= min_area] # ゴミの足切り
            df = df[df['perimeter'] > 0]
            df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
            
            original_w, _ = Image.open(f).size
            current_w, _ = img_raw.size
            scale_factor = original_w / current_w
            df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel * scale_factor
            
            df_clean = df[df['circularity'] > circularity_threshold].copy()
            
            with col_res1:
                fig, ax = plt.subplots()
                ax.imshow(img_np)
                # 輪郭の描画
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
