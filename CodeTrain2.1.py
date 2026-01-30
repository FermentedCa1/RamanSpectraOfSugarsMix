import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# --- THIẾT LẬP THIẾT BỊ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Chiến thần đang chạy trên: {device}")


# ==========================================
# 1. KIẾN TRÚC RESNET 1D (Giữ nguyên vì nó rất bá)
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


class SugarResNet(nn.Module):
    def __init__(self, num_targets=4):
        super(SugarResNet, self).__init__()
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
# 2. HÀM TIỀN XỬ LÝ SIÊU CẤP (Data Cleaning & Prep)
# ==========================================
def prepare_data_v2():
    print("📂 Đang nạp dữ liệu và thanh lọc...")

    # Đọc Spectra (Hàng là Wavenumber, Cột là Sample)
    df_spec = pd.read_csv('Sugar_Concentration_Test_ALL_spectra.csv', index_col=0)
    X_raw = df_spec.T  # Xoay ngang: Hàng là Mẫu

    # Đọc Metadata để lấy nhãn
    df_meta = pd.read_csv('Sugar_Concentrations.xlsx - Sheet1 (2).csv')

    # --- BƯỚC 1: TRUY TÌM VÀ TRẢM MẪU LỖI ---
    # Quét vùng pixel nghi vấn (ví dụ 1400-1650) để tìm 'cột đình' > 2000
    mask_sach = np.all(X_raw.iloc[:, 1400:1650] < 2000, axis=1)
    X_clean_df = X_raw[mask_sach]

    print(f"✅ Đã dọn rác! Loại bỏ {len(X_raw) - len(X_clean_df)} mẫu nhiễu nặng.")

    black_list = ['E3_3', 'E4_3']
    def check_not_outlier(sample_name):
        parts = sample_name.split('_')
        cell_id = f"{parts[4]}_{parts[5]}"
        return cell_id not in black_list

    mask_khong_outlier = [check_not_outlier(name) for name in X_clean_df.index]
    X_clean_df = X_clean_df[mask_khong_outlier]

    print(f"✂️ Đã gặt bỏ thêm các mẫu Outliers từ giếng: {black_list}")
    print(f"📊 Số lượng mẫu còn lại để huấn luyện: {len(X_clean_df)}")

    # --- BƯỚC 2: KHỚP NHÃN (GROUND TRUTH) ---
    target_cols = ['Sucrose [ul]', 'Fructose [ul]', 'Maltose [ul]', 'Glucose [ul]']
    y_list = []
    X_final_list = []

    for sample_name in X_clean_df.index:
        try:
            # Giải mã Cell ID: Sugar_Concentration_Test_54_E6_2_... -> E6_2
            parts = sample_name.split('_')
            cell_id = f"{parts[4]}_{parts[5]}"

            # Lấy nồng độ từ Metadata
            row = df_meta[df_meta['Cell Number'] == cell_id]
            if not row.empty:
                y_list.append(row[target_cols].values[0])
                X_final_list.append(X_clean_df.loc[sample_name].values)
        except:
            continue

    X_final = np.array(X_final_list)
    y_final = np.array(y_list)

    # --- BƯỚC 3: TIỀN XỬ LÝ QUANG PHỔ ---
    # 1. Lọc gai (Median filter)
    X_despiked = median_filter(X_final, size=(1, 3))

    # 2. Tính đạo hàm (Savgol)
    d1 = savgol_filter(X_despiked, window_length=15, polyorder=3, deriv=1)
    d2 = savgol_filter(X_despiked, window_length=15, polyorder=3, deriv=2)

    # 3. Chuẩn hóa SNV
    def snv(data):
        return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)

    # Gộp 3 kênh và chuẩn hóa Y về dải 0-1 (375ul là Max)
    X_processed = np.stack([snv(X_despiked), snv(d1), snv(d2)], axis=1)
    y_scaled = y_final / 375.0

    print(f"📊 Dataset sạch: {len(X_processed)} mẫu. Sẵn sàng luyện công!")
    return train_test_split(X_processed, y_scaled, test_size=0.15, random_state=42), y_final


# ==========================================
# 3. HUẤN LUYỆN CHIẾN THẦN
# ==========================================
(X_train, X_test, y_train, y_test_scaled), y_original = prepare_data_v2()

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = SugarResNet().to(device)
criterion = nn.L1Loss()  # Dùng L1 (MAE) để xử lý Bias tốt hơn MSE
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

print("\n--- Bắt đầu luyện tập ResNet v2.0 ---")
for epoch in range(120):  # Tăng lên tí cho chín
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"🔥 Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']}")

# ==========================================
# 4. ĐÁNH GIÁ (QUY ĐỔI NGƯỢC VỀ UL)
# ==========================================
model.eval()
with torch.no_grad():
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds_scaled = model(X_test_torch).cpu().numpy()

    # Quy đổi ngược về đơn vị ul
    preds_ul = preds_scaled * 375.0
    y_test_ul = y_test_scaled * 375.0

# Vẽ biểu đồ so sánh tổng hợp
sugar_names = ['Sucrose', 'Fructose', 'Maltose', 'Glucose']
plt.figure(figsize=(18, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    r_val = np.corrcoef(y_test_ul[:, i], preds_ul[:, i])[0, 1]
    bias = np.mean(preds_ul[:, i] - y_test_ul[:, i])

    plt.scatter(y_test_ul[:, i], preds_ul[:, i], alpha=0.4, color='darkorange')
    plt.plot([0, 150], [0, 150], 'k--', lw=1)
    plt.title(f"{sugar_names[i]}\nR={r_val:.3f} | Bias={bias:+.2f}")
    plt.xlabel("Thực tế (ul)")
    plt.ylabel("Dự đoán (ul)")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Lưu Model "Hịn"
torch.save(model.state_dict(), 'raman_resnet_v2.pth')
print("\n✅ Đã lưu model v2.0. Đại ca mang sang App dùng ngay cho nóng!")