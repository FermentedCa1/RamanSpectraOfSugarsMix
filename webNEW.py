import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import re

# ==========================================
# 1. KIẾN TRÚC MẠNG RESNET (Giữ nguyên v2.1)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))

class RamanResNet(nn.Module):
    def __init__(self, num_targets=4):
        super(RamanResNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_targets)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)

# ==========================================
# 2. HÀM TIỀN XỬ LÝ (Đồng bộ v2.1)
# ==========================================
def preprocess_input(spectrum):
    clean = median_filter(spectrum, size=3)
    x = clean.reshape(1, -1)
    d1 = savgol_filter(x, window_length=15, polyorder=3, deriv=1)
    d2 = savgol_filter(x, window_length=15, polyorder=3, deriv=2)
    def snv(data):
        return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    x_proc = np.stack([snv(x), snv(d1), snv(d2)], axis=1)
    return torch.tensor(x_proc, dtype=torch.float32)

# ==========================================
# 3. CẤU HÌNH & LOAD DATA
# ==========================================
st.set_page_config(page_title="Raman Analyzer Pro v2.2", layout="wide")
MODEL_PATH = 'raman_resnet_experiment.pth'
METADATA_PATH = 'Sugar_Concentrations.csv'

@st.cache_resource
def load_model():
    model = RamanResNet(num_targets=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_meta():
    return pd.read_csv(METADATA_PATH)

try:
    model = load_model()
    df_meta = load_meta()
except Exception as e:
    st.error(f"⚠️ Lỗi hệ thống: {e}")
    st.stop()

# ==========================================
# 4. SIDEBAR - BỘ LỌC THÔNG MINH
# ==========================================
st.sidebar.header("🛠 Điều khiển & Tìm kiếm")
uploaded_file = st.sidebar.file_uploader("1. Tải file Spectra (.csv)", type="csv")

selected_sample = None

if uploaded_file:
    df_spec = pd.read_csv(uploaded_file)
    all_samples = df_spec.columns[1:].tolist()
    
    tab_search, tab_list = st.sidebar.tabs(["🔍 Tìm theo Giếng", "📋 Danh sách gốc"])
    
    with tab_list:
        selected_sample = st.selectbox("Chọn từ danh sách cuộn:", all_samples)

    with tab_search:
        # Phân tách tên mẫu để tạo bộ lọc (Regex để bắt E4_3, v.v.)
        # Tên mẫu: Sugar_Concentration_Test_52_E4_3_RD1_M1_R2
        try:
            # Lấy danh sách Plate duy nhất
            plates = sorted(list(set([s.split('_')[5] for s in all_samples])))
            sel_plate = st.selectbox("Chọn Plate:", plates)
            
            # Lọc các mẫu thuộc Plate đó
            plate_samples = [s for s in all_samples if s.split('_')[5] == sel_plate]
            
            # Lấy danh sách Hàng (A-H)
            rows = sorted(list(set([re.findall(r'[A-Z]', s.split('_')[4])[0] for s in plate_samples])))
            sel_row = st.select_slider("Chọn Hàng (Row):", options=rows)
            
            # Lọc theo hàng
            row_samples = [s for s in plate_samples if s.split('_')[4].startswith(sel_row)]
            
            # Lấy danh sách Cột (1-12)
            cols = sorted(list(set([int(re.findall(r'\d+', s.split('_')[4])[0]) for s in row_samples])))
            sel_col = st.selectbox("Chọn Cột (Column):", cols)
            
            # Lấy lần lặp (Round/Rep)
            final_options = [s for s in row_samples if s.split('_')[4] == f"{sel_row}{sel_col}"]
            
            if final_options:
                selected_sample = st.radio("Chọn lần đo (Replicates):", final_options)
            else:
                st.warning("Không tìm thấy mẫu phù hợp.")
        except:
            st.error("Cấu trúc tên file không khớp với bộ lọc thông minh.")

# ==========================================
# 5. HIỂN THỊ KẾT QUẢ (Như cũ nhưng ổn định hơn)
# ==========================================
if uploaded_file and selected_sample:
    spectrum = df_spec[selected_sample].values
    wavenumbers = df_spec.iloc[:, 0].values

    st.title(f"🔬 Phân tích mẫu: {selected_sample}")
    col_plot, col_res = st.columns([1.3, 1])

    with col_plot:
        st.subheader("📈 Đồ thị phổ Raman")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavenumbers, spectrum, color='lightgray', lw=1, label='Raw', alpha=0.5)
        clean = median_filter(spectrum, size=3)
        ax.plot(wavenumbers, clean, color='#008080', lw=1.5, label='Median Filtered')
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Intensity")
        ax.legend()
        st.pyplot(fig)

    with col_res:
        st.subheader("📊 Kết quả AI vs Metadata")
        input_tensor = preprocess_input(spectrum)
        with torch.no_grad():
            preds = np.maximum(model(input_tensor).numpy()[0] * 375.0, 0)
        
        sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]
        target_cols = [f'{s} [ul]' for s in sugars]
        
        parts = selected_sample.split('_')
        cell_id = f"{parts[4]}_{parts[5]}"
        truth_row = df_meta[df_meta['Cell Number'] == cell_id]

        if not truth_row.empty:
            actuals = truth_row[target_cols].values[0]
            compare_df = pd.DataFrame({
                "Thành phần": sugars,
                "Thực tế": np.round(actuals, 2),
                "AI Dự đoán": np.round(preds, 2),
                "Lệch": np.round(preds - actuals, 2)
            })
            st.table(compare_df)
            st.success(f"💎 MAE: {np.mean(np.abs(preds-actuals)):.2f} µl")
        else:
            for s, p in zip(sugars, preds):
                st.metric(s, f"{p:.2f} µl")

    # Bảng Metrics hiệu năng v2.1
    with st.expander("📝 Thông số hiệu năng hệ thống (Model v2.1)"):
            metrics = pd.DataFrame({
                    "Đường": sugars,
                    "MAE (ul)": [1.90, 2.15, 2.20, 3.38], 
                    "Correlation (R)": [0.9985, 0.9982, 0.9980, 0.9954], 
                    "Trạng thái": ["Ổn định", "Nhạy", "Ổn định", "Rất tốt"]
else:
    st.info("👋 Chào đại ca! Hãy tải file CSV lên để trải nghiệm bộ lọc tìm kiếm mới.")




