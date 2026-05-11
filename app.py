import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps

st.set_page_config(page_title="Sphere Analyzer v4.5", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.5 (スケール確定版)")

with st.sidebar:
    st.header("1. 基本設定 (確定スケール)")
    mag = st.radio("倍率:", ("4x", "10x", "20x", "カスタム"), index=0)
    
    # ImageJでの実測値に基づく正確なスケーリング
    if mag == "4x":
        um_per_pixel = 1.9109
    elif mag == "10x":
        um_per_pixel = 0.7643
    elif mag == "20x":
        um_per_pixel = 0.3817
    else:
        um_per_pixel = st.number_input("μm/px を手動入力", value=1.0, format="%.4f")

    st.header("2. 画像補正")
    invert_image = st.checkbox("画像を白黒反転する", value=True)
    contrast = st.slider("コントラスト", 0.1, 3.0, 0.67)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 0.0)

    st.header("3. 二値化設定")
    extraction_mode = st.radio("抽出方法:", ("背景を一気にくり抜く", "細かい影を拾う"), index=0)
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 0.0)
    
    if extraction_mode == "背景を一気にくり抜く":
        global_offset = st.slider("背景の判定ライン", -0.30, 0.30, -0.13, step=0.01)
    else:
        block_size_slider = st.slider("Block Size", 11, 201, 51, step=10)
        local_offset = st.slider("Offset", -0.10, 0.10, 0.00, step=0.01)

    st.header("4. ノイズ処理")
    remove_white = st.slider("⚪ 白ゴミ取り", 0, 1000, 10, step=10)
    remove_black = st.slider("⚫ 黒穴埋め", 0, 200, 20, step=1)

    st.header("5. 切り離し (Watershed)")
    min_dist = st.number_input("中心間の最小距離 (px)", value=30)

    st.header("6. フィルタ")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_files = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        img_raw = Image.open(f).convert('RGB')
        # オリジナルのサイズを保持（計測精度のため）
        original_w, original_h = img_raw.size
        
        # プレビュー用にリサイズ
        preview_img = img_raw.copy()
        preview_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        current_w, current_h = preview_img.size
        scale_factor = original_w / current_w
        
        img_edit = preview_img.copy()
        if invert_image:
            img_edit = ImageOps.invert(img_edit)
        img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
        img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
        
        img_np = np.array(img_edit)
        gray = color.rgb2gray(img_np)
        
        if blur_sigma > 0:
            blurred = filters.gaussian(gray, sigma=blur_sigma)
        else:
            blurred = gray

        if extraction_mode == "背景を一気にくり抜く":
            base_thresh = filters.threshold_otsu(blurred)
            adj_thresh = base_thresh + global_offset
            binary = blurred > adj_thresh if invert_image else blurred < adj_thresh
        else:
            block_size = int(block_size_slider)
            if block_size % 2 == 0: block_size += 1
            local_thresh = filters.threshold_local(blurred, block_size, offset=local_offset)
            binary = blurred > local_thresh if invert_image else blurred < local_thresh 
            
        cleaned = morphology.remove_small_objects(binary, min_size=remove_white) if remove_white > 0 else binary
        filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black) if remove_black > 0 else cleaned
        
        distance = ndi.distance_transform_edt(filled)
        coords = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filled)
        
        if exclude_border:
            labels = segmentation.clear_border(labels)

        st.subheader(f"解析: {f.name}")
        
        props = measure.regionprops(labels)
        df_list = []

        for p in props:
            area = p.area
            perimeter = p.perimeter
            if perimeter > 0 and area >= min_area:
                circ = (4 * np.pi * area) / (perimeter ** 2)
                if circ > circularity_threshold:
                    # 面積から逆算した直径(Equivalent Diameter)をμmに変換
                    diam_um = p.equivalent_diameter * um_per_pixel * scale_factor
                    df_list.append({
                        'label': p.label,
                        'equivalent_diameter_um': diam_um,
                        'circularity': circ,
                        'centroid_y': p.centroid[0],
                        'centroid_x': p.centroid[1],
                        'area_px': area
                    })
        
        df_base = pd.DataFrame(df_list)
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res2:
            show_line = st.checkbox("🟢 緑の線を表示する", value=True)
            if not df_base.empty:
                display_count = len(df_base)
                display_diameter = df_base['equivalent_diameter_um'].mean()
                display_circularity = df_base['circularity'].mean()

                st.metric("検出数", f"{display_count} 個")
                st.metric("平均直径 (面積逆算)", f"{display_diameter:.2f} μm")
                st.metric("平均真円度", f"{display_circularity:.2f}")
                
                # CSV出力用にデータを整理
                csv_df = df_base[['label', 'equivalent_diameter_um', 'circularity', 'area_px']].copy()
                csv_df.columns = ['ID', 'Diameter(um)', 'Circularity', 'Area(px)']
                csv = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button("結果をCSVで保存", csv, f"result_{f.name}.csv", "text/csv")
            else:
                st.warning("検出されませんでした。")
                
        with col_res1:
            fig, ax = plt.subplots()
            ax.imshow(img_np) 
            if not df_base.empty and show_line:
                valid_labels = df_base['label'].values
                valid_mask = np.isin(labels, valid_labels)
                filtered_labels = np.where(valid_mask, labels, 0)
                if filtered_labels.max() > 0:
                    ax.contour(filtered_labels, levels=np.arange(filtered_labels.max() + 1) + 0.5, 
                               colors='lime', linewidths=0.5)
            ax.axis('off')
            st.pyplot(fig)
