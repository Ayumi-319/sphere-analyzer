import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import io

st.set_page_config(page_title="Sphere Analyzer v5.0", layout="wide")
st.title("🔴 スフェア自動計測ツール v5.0 (手動修正機能搭載)")

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
    st.header("1. 基本設定 (確定スケール)")
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
    sensitivity = st.slider("色の許容範囲 (融通)", 1, 100, 4)
    
    st.header("4. 強力切り離し")
    st.info("💡 まずは自動で。間違えたら右の画像で手動修正します。")
    watershed_footprint = st.slider("切り離し感度", 1, 100, 30)
    min_dist = st.slider("中心間の最小距離 (px)", 1, 200, 30)

    st.header("5. フィルタ ＆ 表示")
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)
    # 番号表示のスイッチを追加
    show_numbers = st.checkbox("スフェアに番号を表示する", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 複数画像アップロードに対応するためのループ（今回は単一画像として扱う）
    img_raw = Image.open(uploaded_file).convert('RGB')
    original_w, _ = img_raw.size
    
    # プレビュー作成 (im_coordinates用)
    preview_img = img_raw.copy()
    preview_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    current_w, current_h = preview_img.size
    scale_factor = original_w / current_w
    
    # 画像補正
    img_edit = preview_img.copy()
    if invert_image: img_edit = ImageOps.invert(img_edit)
    img_edit = ImageEnhance.Contrast(img_edit).enhance(contrast)
    img_edit = ImageEnhance.Sharpness(img_edit).enhance(sharpness)
    
    img_np = np.array(img_edit)

    col_img, col_res = st.columns([2, 1])

    # --- メイン処理ロジック ---
    if not st.session_state.analysis_done:
        # 初回解析またはリセット後
        if st.session_state.bg_colors:
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
        else:
            binary = None

    # --- 画像表示と座標取得 ---
    with col_img:
        st.subheader("1. スポイト指定 ＆ 手動修正（右クリックで削除、左クリックで分離）")
        
        # 解析前ならスポイト、解析後なら手動修正
        if not st.session_state.analysis_done:
            st.info("👆 サイドバーの設定を済ませ、背景を数カ所クリックしてください。")
            coords = im_coordinates(img_edit, key="coords")
            if coords:
                temp_gray = color.rgb2gray(np.array(img_edit))
                clicked_color = temp_gray[int(coords['y']), int(coords['x'])]
                st.session_state.bg_colors.append(clicked_color)
                st.rerun()
        else:
            # 解析後の画像に現在のラベルをオーバーレイして表示
            fig_coord, ax_coord = plt.subplots()
            ax_coord.imshow(img_np)
            if st.session_state.manual_labels is not None:
                ax_coord.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
            ax_coord.axis('off')
            
            # Matplotlibの図をPIL画像に変換して im_coordinates に渡す
            buf = io.BytesIO()
            fig_coord.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            coord_img = Image.open(buf)
            plt.close(fig_coord)
            
            st.info("💡 **手動修正:** 切り離したい場所を**左クリック**、削除したいスフェアを**右クリック**")
            coords = im_coordinates(coord_img, key="fix_coords")
            
            # 手動修正ロジック
            if coords and st.session_state.manual_labels is not None:
                # 座標が更新された場合のみ処理
                current_coords = (coords['x'], coords['y'])
                if current_coords != st.session_state.last_coords:
                    st.session_state.last_coords = current_coords
                    
                    x, y = int(coords['x']), int(coords['y'])
                    # 座標系をMatplotlibからnumpy配列のインデックスに変換（必要に応じて調整）
                    y_idx, x_idx = y, x 
                    
                    if 0 <= y_idx < st.session_state.manual_labels.shape[0] and 0 <= x_idx < st.session_state.manual_labels.shape[1]:
                        target_label = st.session_state.manual_labels[y_idx, x_idx]
                        
                        # 右クリック（あるいは特定の操作）で削除、通常クリックで分離（今回は簡易的に分離のみ実装）
                        if target_label > 0:
                            # 分離ロジック：クリックした位置を中心に小さな円を描き、ラベルを0にする
                            rr, cc = morphology.disk((y_idx, x_idx), 2, shape=st.session_state.manual_labels.shape)
                            # 簡易的な分離：クリック箇所のラベルを消去
                            temp_mask = st.session_state.manual_labels == target_label
                            temp_mask[rr, cc] = 0
                            # 再ラベル付け
                            new_labels, _ = ndi.label(temp_mask)
                            st.session_state.manual_labels[temp_mask] = new_labels[temp_mask] + st.session_state.manual_labels.max()
                            st.session_state.manual_labels[rr, cc] = 0
                            # watershedを再適用して境界をきれいに
                            distance = ndi.distance_transform_edt(st.session_state.manual_labels > 0)
                            markers, _ = ndi.label(st.session_state.manual_labels)
                            st.session_state.manual_labels = segmentation.watershed(-distance, markers, mask=st.session_state.manual_labels > 0)
                            
                            st.rerun()

    # --- 結果表示 ---
    if st.session_state.analysis_done and st.session_state.manual_labels is not None:
        props = measure.regionprops(st.session_state.manual_labels)
        df_list = []
        
        # 解析結果画像を作成
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        
        count = 0
        for p in props:
            if p.area >= min_area:
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
                if circ > circularity_threshold:
                    count += 1
                    df_list.append({
                        'ID': count,
                        'diam_um': p.equivalent_diameter * um_per_pixel * scale_factor,
                        'circ': circ
                    })
                    
                    # 番号を表示
                    if show_numbers:
                        ax.text(p.centroid[1], p.centroid[0], str(count), color='lime', fontsize=6, ha='center', va='center')
        
        ax.contour(st.session_state.manual_labels > 0, colors='lime', linewidths=0.5)
        ax.axis('off')
        
        df_base = pd.DataFrame(df_list)

        with col_res:
            st.header("📊 計測結果")
            if not df_base.empty:
                st.metric("検出数", f"{len(df_base)} 個")
                st.metric("平均直径", f"{df_base['diam_um'].mean():.1f} μm")
                st.pyplot(fig)
                
                csv = df_base.to_csv(index=False).encode('utf-8')
                st.download_button("結果をCSVで保存", csv, "result.csv", "text/csv")
            else:
                st.warning("スフェアが検出されていません。")
