import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

st.set_page_config(page_title="Sphere Analyzer v4.6", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.6 (背景スポイト抽出)")

with st.sidebar:
    st.header("1. 基本設定 (確定スケール)")
    mag = st.radio("倍率:", ("4x", "10x", "20x", "カスタム"), index=0)
    
    if mag == "4x":
        um_per_pixel = 1.9109
    elif mag == "10x":
        um_per_pixel = 0.7643
    elif mag == "20x":
        um_per_pixel = 0.3817
    else:
        um_per_pixel = st.number_input("μm/px を手動入力", value=1.0, format="%.4f")

    st.header("2. 二値化 (抽出モード選択)")
    # 新機能：スポイト抽出モードを追加
    extraction_mode = st.radio("抽出方法:", ("スポイトで背景色を指定する [新機能]", "背景を一気にくり抜く (従来)", "細かい影を拾う (従来)"), index=0)
    
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 1.0)
    
    if extraction_mode == "スポイトで背景色を指定する [新機能]":
        st.info("👈 右の画像で背景（何もない場所）をクリックしてください")
        # セッション状態にクリック座標を保存
        if 'bg_point' not in st.session_state:
            st.session_state.bg_point = None
            
        sensitivity = st.slider("抽出の感度", 1, 100, 30, help="数値を上げると、指定した背景色に近い色までスフェアとして拾います")
        
    elif extraction_mode == "背景を一気にくり抜く (従来)":
        global_offset = st.slider("背景の判定ライン", -0.30, 0.30, -0.13, step=0.01)
    else:
        block_size_slider = st.slider("Block Size", 11, 201, 51, step=10)
        local_offset = st.slider("Offset", -0.10, 0.10, 0.00, step=0.01)

    st.header("3. ノイズ処理")
    remove_white = st.slider("⚪ 白ゴミ取り", 0, 1000, 10, step=10)
    remove_black = st.slider("⚫ 黒穴埋め", 0, 200, 20, step=1)

    st.header("4. 切り離し (Watershed)")
    min_dist = st.number_input("中心間の最小距離 (px)", value=30)

    st.header("5. フィルタ")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_files = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'], accept_multiple_files=False)

if uploaded_files:
    f = uploaded_files
    img_raw = Image.open(f).convert('RGB')
    
    original_w, original_h = img_raw.size
    
    preview_img = img_raw.copy()
    preview_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    current_w, current_h = preview_img.size
    scale_factor = original_w / current_w
    
    col_img, col_res = st.columns([2, 1])
    
    with col_img:
        st.subheader("解析・スポイト指定")
        # 画像を表示し、クリック座標を取得するコンポーネント
        coords = im_coordinates(preview_img, key="coords")
        
        if coords and extraction_mode == "スポイトで背景色を指定する [新機能]":
            st.session_state.bg_point = (coords['x'], coords['y'])
            st.success(f"背景色を指定しました: ({coords['x']}, {coords['y']})")
    
    # ぼかし処理
    img_np = np.array(preview_img)
    gray = color.rgb2gray(img_np)
    if blur_sigma > 0:
        gray = filters.gaussian(gray, sigma=blur_sigma)

    # --- 二値化ロジックの分岐 ---
    binary = None
    
    if extraction_mode == "スポイトで背景色を指定する [新機能]":
        if st.session_state.bg_point:
            x, y = st.session_state.bg_point
            # プレビュー画像の指定座標から背景色を取得
            bg_color = gray[int(y), int(x)]
            
            # 背景色との差分を計算
            diff = np.abs(gray - bg_color)
            
            # 感度に基づいて二値化（差が大きい部分をスフェアとする）
            thresh = sensitivity / 255.0
            binary = diff > thresh
            
        else:
            st.warning("画像をクリックして背景色を指定してください。")
            st.stop()
            
    elif extraction_mode == "背景を一気にくり抜く (従来)":
        base_thresh = filters.threshold_otsu(gray)
        adj_thresh = base_thresh + global_offset
        # 背景を暗く（反転）する前提
        binary = gray < adj_thresh
        
    else:
        block_size = int(block_size_slider)
        if block_size % 2 == 0: block_size += 1
        local_thresh = filters.threshold_local(gray, block_size, offset=local_offset)
        binary = gray < local_thresh 
        
    # --- 以降の処理は共通 ---
    if binary is not None:
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white) if remove_white > 0 else binary
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black) if remove_black > 0 else cleaned
        
        distance = ndi.distance_transform_edt(filled)
        coords_peak = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords_peak.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        
        if exclude_border:
            labels = segmentation.clear_border(labels)

        props = measure.regionprops(labels)
        df_list = []

        for p in props:
            area = p.area
            perimeter = p.perimeter
            if perimeter > 0 and area >= min_area:
                circ = (4 * np.pi * area) / (perimeter ** 2)
                if circ > circularity_threshold:
                    diam_um = p.equivalent_diameter * um_per_pixel * scale_factor
                    df_list.append({
                        'label': p.label,
                        'equivalent_diameter_um': diam_um,
                        'circularity': circ,
                        'centroid_y': p.centroid[0],
                        'centroid_x': p.centroid[1]
                    })
        
        df_base = pd.DataFrame(df_list)
        
        with col_res:
            st.header("📊 結果・表示設定")
            show_line = st.checkbox("🟢 緑の線を表示する", value=True)
            
            if not df_base.empty:
                display_count = len(df_base)
                display_diameter = df_base['equivalent_diameter_um'].mean()
                display_circularity = df_base['circularity'].mean()

                st.metric("検出数", f"{display_count} 個")
                st.metric("平均直径 (面積逆算)", f"{display_diameter:.2f} μm")
                st.metric("平均真円度", f"{display_circularity:.2f}")
                
                # CSV出力用にデータを整理
                csv_df = df_base[['equivalent_diameter_um', 'circularity']].copy()
                csv_df.columns = ['Diameter(um)', 'Circularity']
                csv = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button("結果をCSVで保存", csv, f"result_{f.name}.csv", "text/csv")
            else:
                st.warning("検出されませんでした。")

            # 解析結果画像を下に表示
            if show_line and labels is not None:
                fig, ax = plt.subplots()
                # 元画像を表示
                ax.imshow(img_np) 
                valid_labels = df_base['label'].values if not df_base.empty else []
                valid_mask = np.isin(labels, valid_labels)
                filtered_labels = np.where(valid_mask, labels, 0)
                if filtered_labels.max() > 0:
                    ax.contour(filtered_labels, levels=np.arange(filtered_labels.max() + 1) + 0.5, 
                               colors='lime', linewidths=1.0)
                ax.axis('off')
                st.pyplot(fig)
