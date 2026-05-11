import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

st.set_page_config(page_title="Sphere Analyzer v4.7", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.7 (複数スポイト抽出)")

# セッション状態の初期化
if 'bg_colors' not in st.session_state:
    st.session_state.bg_colors = []

with st.sidebar:
    st.header("1. 基本設定 (スケール)")
    mag = st.radio("倍率:", ("4x", "10x", "20x", "カスタム"), index=0)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. 画像補正 (背景を整える)")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 0.67)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 0.0)

    st.header("3. 二値化 (スポイト設定)")
    extraction_mode = st.radio("抽出方法:", ("複数点スポイト [推奨]", "背景を一気にくり抜く", "細かい影を拾う"), index=0)
    
    if extraction_mode == "複数点スポイト [推奨]":
        st.write("💡 右の画像で**背景（何もない場所）を数カ所**クリックしてください。")
        if st.button("スポイト履歴をリセット"):
            st.session_state.bg_colors = []
            st.rerun()
        
        sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 30, help="数値を上げると、少し色が違う場所も背景とみなして消去します")
    
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 0.0)

    st.header("4. ノイズ処理")
    remove_white = st.slider("⚪ 白ゴミ取り", 0, 1000, 10, step=10)
    remove_black = st.slider("⚫ 黒穴埋め", 0, 200, 20, step=1)

    st.header("5. 切り離し (Watershed)")
    min_dist = st.number_input("中心間の最小距離 (px)", value=30)

    st.header("6. フィルタ")
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_files = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'], accept_multiple_files=False)

if uploaded_files:
    img_raw = Image.open(uploaded_files).convert('RGB')
    original_w, original_h = img_raw.size
    
    # プレビュー作成
    preview_img = img_raw.copy()
    preview_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    current_w, current_h = preview_img.size
    scale_factor = original_w / current_w
    
    # --- 画像補正の実行 ---
    img_edit = preview_img.copy()
    if invert_image:
        img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    
    col_img, col_res = st.columns([2, 1])
    
    with col_img:
        st.subheader("1. スポイト指定（背景をクリック）")
        coords = im_coordinates(img_edit, key="coords")
        
        if coords and extraction_mode == "複数点スポイト [推奨]":
            # クリックした位置の色（グレースケール値）を保存
            temp_np = np.array(color.rgb2gray(np.array(img_edit)))
            clicked_color = temp_np[int(coords['y']), int(coords['x'])]
            if clicked_color not in st.session_state.bg_colors:
                st.session_state.bg_colors.append(clicked_color)

    # --- 二値化ロジック ---
    img_np = np.array(img_edit)
    gray = color.rgb2gray(img_np)
    if blur_sigma > 0:
        gray = filters.gaussian(gray, sigma=blur_sigma)

    binary = None
    if extraction_mode == "複数点スポイト [推奨]":
        if st.session_state.bg_colors:
            # 登録された全背景色との差分の最小値をとる
            diff_map = np.ones_like(gray) * 100.0
            for bg_c in st.session_state.bg_colors:
                diff_map = np.minimum(diff_map, np.abs(gray - bg_c))
            
            # 背景色に近い（差が小さい）部分を0、遠い部分を1にする
            binary = diff_map > (sensitivity / 255.0)
        else:
            st.info("画像上の背景部分を数カ所クリックしてください。")
            st.stop()
    elif extraction_mode == "背景を一気にくり抜く":
        base_thresh = filters.threshold_otsu(gray)
        binary = gray > (base_thresh - 0.13) # 簡易実装
    else:
        binary = gray > filters.threshold_local(gray, 51)

    # --- 後処理 ---
    if binary is not None:
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white)
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black)
        
        distance = ndi.distance_transform_edt(filled)
        coords_peak = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords_peak.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        labels = segmentation.clear_border(labels)

        props = measure.regionprops(labels)
        df_list = []
        for p in props:
            if p.area >= min_area:
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
                if circ > circularity_threshold:
                    df_list.append({'label': p.label, 'diam_um': p.equivalent_diameter * um_per_pixel * scale_factor, 'circ': circ})
        df_base = pd.DataFrame(df_list)

        with col_res:
            st.header("📊 計測結果")
            if not df_base.empty:
                st.metric("検出数", f"{len(df_base)} 個")
                st.metric("平均直径", f"{df_base['diam_um'].mean():.2f} μm")
                
                fig, ax = plt.subplots()
                ax.imshow(img_np)
                ax.contour(labels > 0, colors='lime', linewidths=1.0)
                ax.axis('off')
                st.pyplot(fig)
