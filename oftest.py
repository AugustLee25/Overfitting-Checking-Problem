# =====================================================================
# BÀI TOÁN MÔ PHỎNG OVERFITTING & UNDERFITTING (PYTHON)
# Yêu cầu cài đặt thư viện: pip install numpy matplotlib scikit-learn
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_simulation():
    print("Đang khởi tạo dữ liệu...")
    # 1. Tạo dữ liệu giả lập (Hàm sóng sin có nhiễu)
    np.random.seed(42) # Cố định seed để kết quả luôn giống nhau mỗi lần chạy
    
    # Dữ liệu thật (đường cong chuẩn không nhiễu để đối chiếu)
    X_true = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X_true)
    
    # Tập huấn luyện (Train set) - 50 điểm đo đạc có kèm nhiễu ngẫu nhiên
    X_train = np.sort(np.random.rand(50)).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * X_train) + np.random.randn(50).reshape(-1, 1) * 0.2
    
    # Tập kiểm thử (Test set) - 50 điểm mới hoàn toàn để đánh giá khách quan
    X_test = np.sort(np.random.rand(50)).reshape(-1, 1)
    y_test = np.sin(2 * np.pi * X_test) + np.random.randn(50).reshape(-1, 1) * 0.2
    
    # 2. Xây dựng 3 mô hình tương ứng với 3 trạng thái
    degrees = [1, 5, 100]  # Bậc 1: Underfitting, Bậc 3: Good Fit, Bậc 100: Overfitting    
    titles = ['Underfitting (Bậc 1)', 'Good Fit (Bậc 3)', 'Overfitting (Bậc 100)']
    
    plt.figure(figsize=(15, 5))
    
    print("Đang huấn luyện các mô hình...")
    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, 3, i + 1)
        
        # Khởi tạo Pipeline: chuyển đổi X thành đa thức -> đưa vào Hồi quy tuyến tính
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = make_pipeline(polynomial_features, linear_regression)
        
        # Huấn luyện mô hình CHỈ BẰNG DỮ LIỆU TRAIN
        pipeline.fit(X_train, y_train)
        
        # Đánh giá sai số MSE trên cả tập Train và Test
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        # 3. Trực quan hóa kết quả
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        plt.plot(X_plot, pipeline.predict(X_plot), label="Mô hình dự đoán", color='blue', linewidth=2)
        plt.plot(X_true, y_true, label="Hàm thực tế", color='green', linestyle='--')
        
        # Vẽ các điểm dữ liệu
        plt.scatter(X_train, y_train, edgecolor='k', color='blue', s=40, label="Dữ liệu Train")
        plt.scatter(X_test, y_test, edgecolor='k', color='red', s=40, marker='s', label="Dữ liệu Test")
        
        # Căn chỉnh hiển thị
        plt.xlabel("Trục X")
        plt.ylabel("Trục Y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        
        # Đặt tiêu đề hiển thị rõ chỉ số toán học
        plt.title(f"{titles[i]}\nMSE Train: {mse_train:.3f} | MSE Test: {mse_test:.3f}", fontsize=12, fontweight='bold')
        
        if i == 0:
            plt.legend(loc="lower left")
        
    plt.tight_layout()
    
    # Lưu kết quả thành ảnh
    plt.savefig('result_plot.png', dpi=300)
    print("=> Hoàn tất! Đã lưu biểu đồ vào file 'result_plot.png' tại thư mục hiện tại.")

if __name__ == '__main__':
    run_simulation()