import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import io

st.set_page_config(page_title="Sphere Analyzer v5.6", layout="wide")
st.title("🔴 スフェア自動計測ツール v5.6")

if 'bg_colors' not in st.session_state: st.session_state.bg_colors = []
if 'manual_labels' not in st.session_state: st.session_state.manual_labels = None
if 'last_coords' not in st.session_state: st.session_state.last_coords = None

with st.sidebar:
    st.header("🛠️ 操作モード")
    mode = st.radio("現在のモード:", ["🧪 背景スポイト吸い取り", "裁断 ✂️ 手動切り離し・削除"], index=0)
    if st.button("全設定をリセット"):
        st.session_state.bg_colors = []
        st.session_state.manual_labels = None
        st.rerun()

    st.header("1. スケール設定")
    mag = st.radio("倍率:", ("4x", "10x", "20x"), index=1)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 0.90)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0)
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 3.0) # 3.0を推奨

    st.header("3. 二値化設定")
    st.write(f"🎯 スポイト数: **{len(st.session_state.bg_colors)}** 点")
    sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 15)
    
    st.header("4. ゴミ除去・穴埋め")
    remove_white = st.slider("⚪ 白ゴミ取り", 0, 2000, 500, step=50)
    remove_black = st.slider("⚫ 黒穴埋め", 0, 10000, 3000, step=100)

    st.header("5. 切り離し ＆ 強力除去")
    exclude_border = st.checkbox("画像端を除外 (強力モード)", value=True)
    # 端っこの判定幅を調整可能に
    border_buffer = st.slider("端っこの判定幅 (px)", 0, 50, 5, help="端から何px以内にあれば除去するか。少し上げると確実です")
    
    watershed_footprint = st.slider("切り離し感度", 1, 100, 30)
    min_dist = st.slider("最小距離(px)", 1, 200, 30)
    min_area = st.number_input("最小面積(px)", value=1000)
    show_numbers = st.checkbox("番号を表示", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    img_edit = img_raw.copy()
    if invert_image: img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    img_np = np.array(img_edit)

    if mode == "🧪 背景スポイト吸い取り":
        st.subheader("背景をクリックしてください。")
        coords = im_coordinates(img_edit, key="spoid")
        if coords:
            curr = (coords['x'], coords['y'])
            if curr != st.session_state.last_coords:
                st.session_state.last_coords = curr
                gray_temp = color.rgb2gray(img_np)
                st.session_state.bg_colors.append(gray_temp[int(coords['y']), int(coords['x'])])
                st.session_state.manual_labels = None 
                st.rerun()

    if st.session_state.bg_colors and st.session_state.manual_labels is None:
        gray_img = color.rgb2gray(img_np)
        if blur_sigma > 0: gray_img = filters.gaussian(gray_img, sigma=blur_sigma)
        diff_map = np.ones_like(gray_img)
        for bg_c in st.session_state.bg_colors:
            diff_map = np.minimum(diff_map, np.abs(gray_img - bg_c))
        binary = diff_map > (sensitivity / 255.0)
        
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white)
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black)
        
        distance = ndi.distance_transform_edt(filled)
        local_maxi = feature.peak_local_max(distance, min_distance=min_dist, 
                                            footprint=np.ones((watershed_footprint, watershed_footprint)), 
                                            labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        
        if exclude_border:
            # bufferの分だけ内側で判定する強力版除去
            labels = segmentation.clear_border(labels, buffer_size=border_buffer)
            
        st.session_state.manual_labels = labels

    col_img, col_res = st.columns([1.5, 1])

    with col_img:
        # (解析・表示ロジックは5.5を継承)
        fig_v, ax_v = plt.subplots()
        ax_v.imshow(img_np)
        if st.session_state.manual_labels is not None:
            ax_v.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
        ax_v.axis('off')
        st.pyplot(fig_v)
        plt.close(fig_v)

    with col_res:
        # (結果表示ロジックは5.5を継承)
        if st.session_state.manual_labels is not None:
            props = measure.regionprops(st.session_state.manual_labels)
            final_list = []
            fig_r, ax_r = plt.subplots()
            ax_r.imshow(img_np)
            idx = 1
            for p in props:
                if p.area >= min_area:
                    if show_numbers:
                        ax_r.text(p.centroid[1], p.centroid[0], str(idx), color='red', fontsize=8, fontweight='bold', ha='center')
                    final_list.append({'No': idx, '直径(μm)': p.equivalent_diameter * um_per_pixel * (original_w / img_edit.size[0])})
                    idx += 1
            ax_r.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_r.axis('off')
            st.pyplot(fig_r)
            plt.close(fig_r)
            df = pd.DataFrame(final_list)
            st.metric("検出数", f"{len(df)} 個")
            if not df.empty:
                st.download_button("CSV保存", df.to_csv(index=False).encode('utf-8'), "result.csv")
