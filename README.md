Markdown
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
🚀 Hướng dẫn sử dụng
Clone repository này về máy:

Bash
git clone [https://github.com/](https://github.com/)<tên-người-dùng-của-bạn>/<tên-repo>.git
cd <tên-repo>
Chạy file mã nguồn chính:

Bash
python demo_overfitting.py
Xem kết quả:
Sau khi chạy thành công, script sẽ tự động tạo ra một file ảnh tên là result_plot.png ngay trong thư mục chứa code. Mở file ảnh này để xem biểu đồ so sánh 3 trạng thái của mô hình.

📊 Giải thích kết quả
Khi xem file result_plot.png, bạn sẽ thấy 3 đồ thị:

Underfitting (Bậc 1): Mô hình là một đường thẳng. Sai số MSE cao ở cả tập Train và Test. Mô hình quá đơn giản để nắm bắt được quy luật của dữ liệu.

Good Fit (Bậc 3): Mô hình là một đường cong mượt mà, bám sát hàm sóng thực tế. Sai số MSE thấp và đồng đều ở cả hai tập. Đây là trạng thái lý tưởng có khả năng tổng quát hóa tốt.

Overfitting (Bậc 15): Mô hình uốn lượn lắt léo đi qua 100% các điểm Train (MSE Train = 0). Tuy nhiên, khi dự đoán trên tập Test mới, sai số MSE vọt lên khổng lồ. Mô hình đã "học vẹt" nhiễu của dữ liệu cũ và mất đi khả năng dự đoán thực tế.

📝 Giấy phép
Dự án này được tạo ra với mục đích học tập và chia sẻ kiến thức. Bạn có thể tự do sử dụng, chỉnh sửa và phân phối lại!
