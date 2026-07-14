
# Mô phỏng Overfitting và Underfitting trong Machine Learning 🚀

Dự án nhỏ này cung cấp một mã nguồn Python trực quan để minh họa 3 trạng thái cơ bản khi huấn luyện một mô hình Machine Learning: **Underfitting** (Chưa khớp), **Good Fit** (Vừa vặn) và **Overfitting** (Quá khớp). 

Thông qua thuật toán Hồi quy đa thức (Polynomial Regression) và thư viện Scikit-Learn, dự án mô phỏng việc khôi phục một hàm sóng hình sin từ các điểm dữ liệu bị nhiễu.

## 🌟 Tính năng
- Sinh dữ liệu giả lập (hàm sóng sin + nhiễu ngẫu nhiên) chia thành tập Train và Test.
- Huấn luyện 3 mô hình tương ứng với 3 bậc đa thức:
  - **Bậc 1:** Thể hiện Underfitting.
  - **Bậc 3:** Thể hiện Good Fit.
  - **Bậc 15:** Thể hiện Overfitting.
- Tính toán và so sánh trực tiếp chỉ số sai số **MSE (Mean Squared Error)** trên cả tập Train và Test.
- Trực quan hóa kết quả bằng đồ thị Matplotlib và tự động lưu thành file ảnh.

## 🛠️ Yêu cầu môi trường
Để chạy được mã nguồn này, bạn cần cài đặt Python 3.x và các thư viện khoa học dữ liệu cơ bản. 

Sử dụng lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install numpy matplotlib scikit-learn
