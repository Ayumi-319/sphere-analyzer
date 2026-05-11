import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import io

st.set_page_config(page_title="Sphere Analyzer v5.3", layout="wide")
st.title("🔴 スペア自動計測ツール v5.3")

# --- セッション状態の管理 ---
if 'bg_colors' not in st.session_state: st.session_state.bg_colors = []
if 'manual_labels' not in st.session_state: st.session_state.manual_labels = None
if 'last_coords' not in st.session_state: st.session_state.last_coords = None

with st.sidebar:
    st.header("🛠️ 操作モード")
    mode = st.radio("現在のモード:", 
                    ["🧪 背景スポイト吸い取り", "裁断 ✂️ 手動切り離し・削除"], 
                    index=0)
    
    if st.button("全設定をリセット"):
        st.session_state.bg_colors = []
        st.session_state.manual_labels = None
        st.rerun()

    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "20x"), index=1)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 0.92)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.23)

    st.header("3. 二値化設定")
    st.write(f"🎯 スポイト数: **{len(st.session_state.bg_colors)}** 点")
    sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 6)
    
    st.header("4. ゴミ除去・穴埋め")
    remove_white = st.slider("⚪ 白ゴミ取り (背景ノイズ)", 0, 1000, 10, step=10)
    remove_black = st.slider("⚫ 黒穴埋め (内部の穴)", 0, 1000, 20, step=10)

    st.header("5. 切り離し ＆ フィルタ")
    watershed_footprint = st.slider("切り離し感度", 1, 100, 30)
    min_dist = st.slider("最小距離(px)", 1, 200, 30)
    min_area = st.number_input("最小面積(px)", value=100)
    show_numbers = st.checkbox("番号を表示", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    
    # 前処理
    img_edit = img_raw.copy()
    if invert_image: img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    img_np = np.array(img_edit)

    # --- 背景スポイトモード ---
    if mode == "🧪 背景スポイト吸い取り":
        st.subheader("背景（何もない場所）を「1回ずつ丁寧に」クリックしてください")
        coords = im_coordinates(img_edit, key="spoid")
        if coords:
            curr = (coords['x'], coords['y'])
            if curr != st.session_state.last_coords:
                st.session_state.last_coords = curr
                gray_temp = color.rgb2gray(img_np)
                st.session_state.bg_colors.append(gray_temp[int(coords['y']), int(coords['x'])])
                st.session_state.manual_labels = None 
                st.rerun()

    # --- 解析実行 ---
    if st.session_state.bg_colors and st.session_state.manual_labels is None:
        gray_img = color.rgb2gray(img_np)
        diff_map = np.ones_like(gray_img)
        for bg_c in st.session_state.bg_colors:
            diff_map = np.minimum(diff_map, np.abs(gray_img - bg_c))
        binary = diff_map > (sensitivity / 255.0)
        
        # ゴミ除去適用
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white) if remove_white > 0 else binary
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black) if remove_black > 0 else cleaned
        
        distance = ndi.distance_transform_edt(filled)
        local_maxi = feature.peak_local_max(distance, min_distance=min_dist, 
                                            footprint=np.ones((watershed_footprint, watershed_footprint)), 
                                            labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(mask)
        st.session_state.manual_labels = segmentation.watershed(-distance, markers, mask=filled)

    # --- メイン表示 ＆ 裁断モード ---
    col_img, col_res = st.columns([1.5, 1])

    with col_img:
        if mode == "裁断 ✂️ 手動切り離し・削除" and st.session_state.manual_labels is not None:
            st.subheader("くびれ部分を「1回クリック」で分離 / スペア上を「右クリック」で削除")
            # 修正用の描画
            fig_f, ax_f = plt.subplots()
            ax_f.imshow(img_np)
            ax_f.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_f.axis('off')
            
            buf = io.BytesIO()
            fig_f.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            fix_img = Image.open(buf)
            plt.close(fig_f)
            
            coords_fix = im_coordinates(fix_img, key="fix")
            if coords_fix:
                curr_f = (coords_fix['x'], coords_fix['y'])
                if curr_f != st.session_state.last_coords:
                    st.session_state.last_coords = curr_f
                    
                    data_h, data_w = st.session_state.manual_labels.shape
                    tx = int(coords_fix['x'] * data_w / fix_img.size[0])
                    ty = int(coords_fix['y'] * data_h / fix_img.size[1])
                    
                    if 0 <= ty < data_h and 0 <= tx < data_w:
                        l_val = st.session_state.manual_labels[ty, tx]
                        if l_val > 0:
                            # 1点クリックで微小な穴を開けて再分割
                            # エラー防止のため座標チェック
                            rr, cc = morphology.disk((ty, tx), 2, shape=(data_h, data_w))
                            st.session_state.manual_labels[rr, cc] = 0
                            remaining = st.session_state.manual_labels == l_val
                            new_l, _ = ndi.label(remaining)
                            st.session_state.manual_labels[remaining] = new_l[remaining] + st.session_state.manual_labels.max()
                            st.rerun()
        else:
            fig_v, ax_v = plt.subplots()
            ax_v.imshow(img_np)
            if st.session_state.manual_labels is not None:
                ax_v.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_v.axis('off')
            st.pyplot(fig_v)
            plt.close(fig_v)

    with col_res:
        if st.session_state.manual_labels is not None:
            props = measure.regionprops(st.session_state.manual_labels)
            final_list = []
            fig_r, ax_r = plt.subplots()
            ax_r.imshow(img_np)
            idx = 1
            for p in props:
                if p.area >= min_area:
                    if show_numbers:
                        ax_r.text(p.centroid[1], p.centroid[0], str(idx), color='red', fontsize=8, fontweight='bold')
                    final_list.append({'No': idx, '直径(μm)': p.equivalent_diameter * um_per_pixel})
                    idx += 1
            ax_r.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_r.axis('off')
            st.pyplot(fig_r)
            plt.close(fig_r)
            df = pd.DataFrame(final_list)
            st.metric("検出数", f"{len(df)} 個")
            if not df.empty:
                st.download_button("CSV保存", df.to_csv(index=False).encode('utf-8'), "result.csv")
