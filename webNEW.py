import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

# ==========================================
# 1. KIẾN TRÚC MẠNG RESNET (Giữ nguyên cấu trúc v2.1)
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
# 2. HÀM TIỀN XỬ LÝ (Chuẩn hóa v2.1)
# ==========================================
def preprocess_input(spectrum):
    # 1. Lọc gai nhiễu
    clean = median_filter(spectrum, size=3)
    x = clean.reshape(1, -1)
    
    # 2. Đạo hàm Savitzky-Golay
    d1 = savgol_filter(x, window_length=15, polyorder=3, deriv=1)
    d2 = savgol_filter(x, window_length=15, polyorder=3, deriv=2)

    # 3. Chuẩn hóa SNV
    def snv(data):
        return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)

    x_proc = np.stack([snv(x), snv(d1), snv(d2)], axis=1)
    return torch.tensor(x_proc, dtype=torch.float32)

# ==========================================
# 3. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Raman Sugar Pro v2.1", layout="wide")

MODEL_PATH = 'raman_resnet_v2.1.pth' # Đường dẫn bản v2.1 đại ca vừa train
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

# Khởi tạo
try:
    model = load_model()
    df_meta = load_meta()
except Exception as e:
    st.error(f"⚠️ Lỗi nạp file: {e}. Đại ca check lại file .pth và .csv nhé!")
    st.stop()

# Giao diện chính
st.title("🔬 Hệ thống Phân tích Đa thành phần Đường v2.1")
st.markdown("---")

# ==========================================
# 4. SIDEBAR ĐIỀU KHIỂN
# ==========================================
st.sidebar.header("📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file Spectra (.csv)", type="csv")

if uploaded_file:
    df_spec = pd.read_csv(uploaded_file)
    all_samples = df_spec.columns[1:].tolist()
    selected_sample = st.sidebar.selectbox("🎯 Chọn mẫu phân tích:", all_samples)
    
    spectrum = df_spec[selected_sample].values
    wavenumbers = df_spec.iloc[:, 0].values

    # ==========================================
    # 5. HIỂN THỊ VÀ DỰ ĐOÁN
    # ==========================================
    col_plot, col_res = st.columns([1.3, 1])

    with col_plot:
        st.subheader(f"📈 Phổ Raman: {selected_sample}")
        fig, ax = plt.subplots(figsize=(10, 5))
        # Vẽ phổ gốc và phổ đã lọc để so sánh độ "sạch"
        ax.plot(wavenumbers, spectrum, color='lightgray', lw=1, label='Raw Signal', alpha=0.5)
        spectrum_clean = median_filter(spectrum, size=3)
        ax.plot(wavenumbers, spectrum_clean, color='#008080', lw=1.5, label='Processed (Median Filter)')
        
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with col_res:
        st.subheader("📊 Kết quả Phân tích AI")
        
        # --- CHẠY AI ---
        input_tensor = preprocess_input(spectrum)
        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()[0]
            # Quy đổi nồng độ (0-1 -> 0-375ul)
            preds = np.maximum(pred_scaled * 375.0, 0)

        # --- ĐỐI CHIẾU METADATA ---
        sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]
        target_cols = ['Sucrose [ul]', 'Fructose [ul]', 'Maltose [ul]', 'Glucose [ul]']
        
        try:
            parts = selected_sample.split('_')
            cell_id = f"{parts[4]}_{parts[5]}"
            truth_row = df_meta[df_meta['Cell Number'] == cell_id]
        except:
            truth_row = pd.DataFrame()

        if not truth_row.empty:
            actuals = truth_row[target_cols].values[0]
            
            # Vẽ biểu đồ so sánh cột
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            x = np.arange(len(sugars))
            width = 0.35
            ax_bar.bar(x - width/2, actuals, width, label='Thực tế (Metadata)', color='#2E8B57', alpha=0.8)
            ax_bar.bar(x + width/2, preds, width, label='AI Dự đoán', color='#CD5C5C', alpha=0.8)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(sugars)
            ax_bar.set_ylabel("Thể tích (µl)")
            ax_bar.legend()
            st.pyplot(fig_bar)

            # Bảng so sánh
            compare_df = pd.DataFrame({
                "Thành phần": sugars,
                "Thực tế": [f"{v:.2f}" for v in actuals],
                "AI Dự đoán": [f"{v:.2f}" for v in preds],
                "Lệch (Error)": [f"{p - a:+.2f}" for a, p in zip(actuals, preds)]
            })
            st.table(compare_df)
            
            mae_sample = np.mean(np.abs(preds - actuals))
            st.info(f"💡 Sai số trung bình của mẫu này: **{mae_sample:.2f} µl**")
        else:
            st.warning(f"⚠️ Không tìm thấy nhãn thực tế cho Cell ID: {cell_id}")
            for s, p in zip(sugars, preds):
                st.metric(label=s, value=f"{p:.2f} µl")

    # ==========================================
    # 6. PHẦN "KHÈ" HỘI ĐỒNG (HIỆU NĂNG V2.1)
    # ==========================================
    st.markdown("---")
    with st.expander("🔬 Thông số kỹ thuật & Hiệu năng mô hình v2.1 (Batch Test Results)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Quy trình xử lý:**")
            st.code("Median Filter (3) -> Savgol (15, 3, deriv=1&2) -> SNV Normalization")
            st.write("**Kiến trúc:** 1D-ResNet với Residual Blocks (Skip Connections)")
        
        with col2:
            st.write("**Đánh giá tổng quát trên tập Test:**")
            # Cập nhật đúng các con số đại ca vừa test xong nhé!
            metrics_data = {
                "Loại đường": sugars,
                "MAE (µl)": [2.77, 2.59, 2.76, 4.41],
                "Correlation (R)": [0.9927, 0.9967, 0.9964, 0.9931]
            }
            st.table(pd.DataFrame(metrics_data))
            st.success("🎯 Sai số trung bình hệ thống: 3.13 µl")

else:
    st.info("👋 Chào đại ca! Vui lòng tải file spectra vào sidebar để bắt đầu soi lỗi và dự đoán nồng độ.")
    # Chèn một cái ảnh sơ đồ kiến trúc cho chuyên nghiệp
