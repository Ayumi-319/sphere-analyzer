import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import io

st.set_page_config(page_title="Sphere Analyzer v5.1", layout="wide")
st.title("🔴 スフェア自動計測ツール v5.1")

# --- セッション状態の初期化 ---
if 'bg_colors' not in st.session_state:
    st.session_state.bg_colors = []
if 'manual_labels' not in st.session_state:
    st.session_state.manual_labels = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'last_coords' not in st.session_state:
    st.session_state.last_coords = None

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "20x"), index=1)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 1.25)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 4.21)

    st.header("3. 二値化 (スポイト)")
    if st.button("スポイト履歴をリセット"):
        st.session_state.bg_colors = []
        st.session_state.analysis_done = False
        st.session_state.manual_labels = None
        st.rerun()
    
    # スポイト数の表示を復活
    st.write(f"✅ 現在のスポイト選択数: **{len(st.session_state.bg_colors)}** 点")
    
    sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 4)
    
    st.header("4. 切り離し ＆ フィルタ")
    watershed_footprint = st.slider("切り離し感度", 1, 100, 30)
    min_dist = st.slider("中心間の最小距離 (px)", 1, 200, 30)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)
    show_numbers = st.checkbox("スフェアに番号を表示する", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    original_w, original_h = img_raw.size
    
    # 画像補正プロセスの適用
    img_edit = img_raw.copy()
    if invert_image: img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    
    img_np = np.array(img_edit)
    col_img, col_res = st.columns([1.5, 1])

    # --- 解析ロジック ---
    if not st.session_state.analysis_done and st.session_state.bg_colors:
        gray_img = color.rgb2gray(img_np)
        diff_map = np.ones_like(gray_img)
        for bg_c in st.session_state.bg_colors:
            diff_map = np.minimum(diff_map, np.abs(gray_img - bg_c))
        binary = diff_map > (sensitivity / 255.0)
        
        cleaned = morphology.remove_small_objects(binary, min_size=10)
        filled = morphology.remove_small_holes(cleaned, area_threshold=10)
        
        distance = ndi.distance_transform_edt(filled)
        local_maxi = feature.peak_local_max(distance, min_distance=min_dist, 
                                            footprint=np.ones((watershed_footprint, watershed_footprint)), 
                                            labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        st.session_state.manual_labels = segmentation.clear_border(labels)
        st.session_state.analysis_done = True

    # --- メイン画像表示エリア ---
    with col_img:
        if not st.session_state.analysis_done:
            st.subheader("1. 背景を数カ所クリックしてください")
            coords = im_coordinates(img_edit, key="pickup")
            if coords:
                temp_gray = color.rgb2gray(img_np)
                # 座標の安全な取得
                y_c, x_c = int(coords['y']), int(coords['x'])
                if 0 <= y_c < temp_gray.shape[0] and 0 <= x_c < temp_gray.shape[1]:
                    clicked_color = temp_gray[y_c, x_c]
                    st.session_state.bg_colors.append(clicked_color)
                    st.rerun()
        else:
            st.subheader("2. 解析 ＆ 手動修正")
            st.info("💡 左クリック：その場所で分割 / 右クリック：その塊を削除")
            
            # プレビュー用の図を作成
            fig_p, ax_p = plt.subplots()
            ax_p.imshow(img_np)
            if st.session_state.manual_labels is not None:
                ax_p.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_p.axis('off')
            
            # 解析後の画像を再描画して座標取得に使う
            buf = io.BytesIO()
            fig_p.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            displayed_img = Image.open(buf)
            plt.close(fig_p)
            
            coords = im_coordinates(displayed_img, key="manual_fix")
            
            if coords:
                curr = (coords['x'], coords['y'])
                if curr != st.session_state.last_coords:
                    st.session_state.last_coords = curr
                    
                    # 表示画像とデータ配列の比率計算（エラー防止）
                    disp_w, disp_h = displayed_img.size
                    data_h, data_w = st.session_state.manual_labels.shape
                    target_x = int(coords['x'] * data_w / disp_w)
                    target_y = int(coords['y'] * data_h / disp_h)
                    
                    if 0 <= target_y < data_h and 0 <= target_x < data_w:
                        label_val = st.session_state.manual_labels[target_y, target_x]
                        if label_val > 0:
                            # 修正ロジック
                            # ここでは「クリックした周囲をわずかに消して再ラベリング」で分割を実現
                            y_idx, x_idx = target_y, target_x
                            # エラー箇所修正：diskの呼び出しを修正
                            rr, cc = morphology.disk((y_idx, x_idx), 2, shape=(data_h, data_w))
                            
                            # 修正処理：一旦そのラベルを消して再計算
                            temp_mask = st.session_state.manual_labels == label_val
                            temp_mask[rr, cc] = 0
                            new_labels, _ = ndi.label(temp_mask)
                            
                            # 全体のラベルを更新
                            st.session_state.manual_labels[st.session_state.manual_labels == label_val] = 0
                            max_l = st.session_state.manual_labels.max()
                            st.session_state.manual_labels[temp_mask] = new_labels[temp_mask] + max_l
                            st.rerun()

    # --- 右側：結果 ＆ 番号付き画像 ---
    with col_res:
        if st.session_state.analysis_done and st.session_state.manual_labels is not None:
            props = measure.regionprops(st.session_state.manual_labels)
            final_data = []
            
            fig_res, ax_res = plt.subplots()
            ax_res.imshow(img_np)
            
            id_count = 1
            for p in props:
                if p.area >= min_area:
                    circ = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
                    if circ > circularity_threshold:
                        # 番号表示
                        if show_numbers:
                            ax_res.text(p.centroid[1], p.centroid[0], str(id_count), 
                                        color='red', fontsize=7, fontweight='bold', ha='center', va='center')
                        
                        final_data.append({
                            'No': id_count,
                            '直径(μm)': p.equivalent_diameter * um_per_pixel,
                            '真円度': circ
                        })
                        id_count += 1
            
            ax_res.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_res.axis('off')
            
            st.pyplot(fig_res)
            df = pd.DataFrame(final_data)
            st.metric("現在の計測数", f"{len(df)} 個")
            st.dataframe(df, height=300)
            
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("結果をCSVで保存", csv, "result_v5.csv")
