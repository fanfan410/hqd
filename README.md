# Báo cáo cuối kỳ – Dự đoán giá nhà (Miami Housing)

## Phần 1: Xử lý dữ liệu

- **1. Mô tả bài toán, đầu vào, đầu ra, yêu cầu xử lý:**  
  - Đầu vào: File CSV (miami-housing.csv) chứa dữ liệu nhà ở Miami.  
  - Đầu ra: Dự đoán giá nhà (SALE_PRC_CLEANED) sau khi tiền xử lý (xử lý outlier, feature engineering, v.v.).  
  - Yêu cầu xử lý: Đọc dữ liệu, khám phá (EDA), xử lý outlier, tạo biến mới, và chuẩn bị dữ liệu cho mô hình hồi quy.  
  (Chi tiết mô tả bài toán, đầu vào, đầu ra, yêu cầu xử lý được ghi trong docstring của hàm explore_data (data_preprocessing.py).)

- **2. Đánh nhãn & Tiền xử lý dữ liệu:**  
  - **Bước đánh nhãn:**  
    - Đọc dữ liệu từ CSV (miami-housing.csv) (hàm load_data) – đầu ra là DataFrame (df) chứa dữ liệu gốc.  
    - (Output) In ra kích thước dữ liệu (df.shape) và dữ liệu mẫu (df.head()).  
    - (Output) In ra thống kê mô tả (df.describe()), kiểu dữ liệu (df.dtypes), số lượng giá trị null (df.isnull().sum()).  
    - (Hình ảnh) Vẽ biểu đồ phân phối giá gốc (price_distribution_original.png) (được vẽ trong hàm explore_data) để mô tả phân phối giá nhà (SALE_PRC) trước khi xử lý outlier.  
    - (Hình ảnh) Vẽ biểu đồ ma trận tương quan (correlation_matrix.png) (được vẽ trong hàm explore_data) để mô tả mối tương quan giữa các biến số (numeric) trong tập dữ liệu.  
  - **Bước tiền xử lý:**  
    - **2.4.1. Xử lý giá trị thiếu:**  
      - (Nên giữ) Xử lý giá trị thiếu (NaN) bằng SimpleImputer (thay bằng giá trị trung bình (cho biến số) hoặc giá trị phổ biến nhất (cho biến phân loại) – đã được thực hiện trong bộ tiền xử lý (hàm create_preprocessor).  
      - (Nên cân nhắc thêm) Nếu số lượng giá trị thiếu không đáng kể, có thể cân nhắc xóa hàng (dropna) (ví dụ, data.dropna(subset=['SALE_PRC'], inplace=True)).  
    - **2.4.2. Xử lý giá trị bất thường (outliers):**  
      - (Nên giữ) Xử lý outlier (hàm handle_outliers) bằng phương pháp IQR (tạo cột SALE_PRC_CLEANED đã clip giá trị) – (Output) In ra số lượng outliers tìm thấy và thông báo đã tạo cột SALE_PRC_CLEANED.  
      - (Hình ảnh) Vẽ biểu đồ (boxplot) so sánh giá nhà trước và sau khi xử lý outlier (outliers_handling.png) (được vẽ trong hàm handle_outliers).  
    - **2.4.3. Chuẩn hóa dữ liệu:**  
      - (Nên giữ) Chuẩn hóa biến số (StandardScaler) – đã được thực hiện trong bộ tiền xử lý (hàm create_preprocessor).  
    - **2.4.4. Mã hóa biến phân loại:**  
      - (Nên giữ) Mã hóa biến phân loại (OneHotEncoder) – đã được thực hiện trong bộ tiền xử lý (hàm create_preprocessor).  
    - **2.4.5. Loại bỏ cột không cần thiết:**  
      - (Nên giữ) Loại bỏ cột không cần thiết (ví dụ, PARCELNO (số hiệu lô đất) chỉ dùng để định danh) – đã được thực hiện trong quy trình chính (dòng 350–360 (drop columns)).  
    - **2.4.6. Chia tập dữ liệu:**  
      - (Nên giữ) Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) (hàm train_test_split) – đã được thực hiện trong quy trình chính (dòng 370–380).  
      (Đoạn code liên quan:  
        – Hàm load_data (đọc CSV) (dòng 40–50)  
        – Hàm explore_data (in thống kê mô tả, kiểu dữ liệu, số lượng giá trị null, vẽ phân phối giá gốc, ma trận tương quan) (dòng 57–80)  
        – Hàm handle_outliers (xử lý outlier bằng IQR, tạo cột SALE_PRC_CLEANED, vẽ outliers_handling.png) (dòng 123–127)  
        – Hàm feature_engineering (tạo biến mới) (dòng 150–160)  
        – Hàm create_preprocessor (tạo bộ tiền xử lý) (dòng 170–190)  
        – Quy trình chính (drop cột không cần thiết, chia tập dữ liệu) (dòng 350–380)  
      )
      - **2.4.7. Tạo biến mới (Feature Engineering):**  
        - (Nên giữ) Tạo các biến mới từ dữ liệu hiện có (hàm feature_engineering):  
          - PRICE_PER_SQFT: Giá trên mỗi foot vuông đất (SALE_PRC / LND_SQFOOT)  
          - LIVING_LAND_RATIO: Tỷ lệ diện tích sống trên diện tích đất (TOT_LVG_AREA / LND_SQFOOT)  
          - AVG_IMPORTANT_DIST: Khoảng cách trung bình đến các điểm quan trọng (trung bình của OCEAN_DIST, WATER_DIST, CNTR_DIST, HWY_DIST)  
        - (Output) In ra thông báo "Đã tạo các biến: PRICE_PER_SQFT, LIVING_LAND_RATIO, AVG_IMPORTANT_DIST."  
    - (Đoạn code liên quan:  
        – Hàm load_data (đọc CSV) (dòng 40–50)  
        – Hàm explore_data (in thống kê mô tả, kiểu dữ liệu, số lượng giá trị null, vẽ phân phối giá gốc, ma trận tương quan) (dòng 57–80)  
        – Hàm handle_outliers (xử lý outlier bằng IQR, tạo cột SALE_PRC_CLEANED, vẽ outliers_handling.png) (dòng 123–127)  )

## Phần 2: Đánh giá mô hình

- So sánh hiệu suất các mô hình cơ bản (Linear, Ridge, Lasso, Random Forest, Gradient Boosting) bằng 5-fold CV (lưu ảnh model_comparison.png).  
- Tinh chỉnh siêu tham số (GridSearchCV) cho mô hình tốt nhất (Gradient Boosting) (lưu mô hình tối ưu vào best_pipeline_optimized.pkl).  
- Vẽ learning curve (lưu ảnh learning_curve.png).  
- Đánh giá mô hình trên tập test (in MAE, RMSE, R², MAPE, lưu ảnh prediction_vs_actual.png, error_distribution.png).  
- Phân tích độ quan trọng của đặc trưng (lưu ảnh feature_importance.png).

## Phần 3: Cải tiến mô hình

- (Yêu cầu của thầy) Đưa ra 8 mô hình để thực hiện cải tiến (ví dụ, SVM, Neural Network, XGBoost, LightGBM, CatBoost, Elastic Net, Decision Tree, hoặc các biến thể khác).  
- So sánh hiệu suất (RMSE, MAE, R², v.v.) giữa các mô hình (vẽ biểu đồ so sánh).  
- (Nếu cần) Tinh chỉnh siêu tham số cho từng mô hình (hoặc mô hình tốt nhất) để cải thiện hiệu suất.

## Phần 4: Đóng gói chương trình

- (Yêu cầu của thầy) Đóng gói chương trình (ví dụ, tạo một gói Python, hoặc một ứng dụng (CLI, GUI, Web) để người dùng có thể dễ dàng sử dụng).  
- (Nếu cần) Viết hướng dẫn sử dụng (hoặc tài liệu) để người dùng có thể chạy chương trình, nhập dữ liệu, và nhận kết quả dự đoán.

---

*Lưu ý: Chi tiết mô tả bài toán, đầu vào, đầu ra, yêu cầu xử lý (Phần 1) được ghi trong docstring của hàm explore_data (data_preprocessing.py).* 