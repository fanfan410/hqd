import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import time  # Để đo thời gian

warnings.filterwarnings('ignore')

# --- Cấu hình ---
DATA_FILE = 'miami-housing.csv'
TARGET_COLUMN = 'SALE_PRC_CLEANED'
ID_COLUMN = 'PARCELNO'
ORIGINAL_TARGET = 'SALE_PRC'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
GRID_SEARCH_CV_FOLDS = 3
N_JOBS = -1  # Sử dụng tất cả CPU cores

# Danh sách các mô hình cơ bản để so sánh
BASE_MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE)
}

# Lưới tham số cho GridSearchCV (cho Gradient Boosting)
GB_PARAM_GRID = {
    # Tiền tố 'model__' là bắt buộc khi dùng GridSearchCV với Pipeline
    'model__n_estimators': [100, 200], # Giảm bớt để chạy nhanh hơn
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5],
    'model__min_samples_split': [5, 10],
    'model__subsample': [0.8, 1.0]
}

# --- Hàm Hỗ trợ ---

def load_data(filepath):
    """Đọc dữ liệu từ file CSV."""
    print(f"\n1. ĐỌC DỮ LIỆU từ {filepath}")
    df = pd.read_csv(filepath)
    print(f"Kích thước dữ liệu: {df.shape}")
    print("\nDữ liệu mẫu:")
    print(df.head())
    return df

def explore_data(df):
    """Thực hiện khám phá dữ liệu cơ bản (EDA)."""
    print("\n2. KHÁM PHÁ DỮ LIỆU (EDA)")
    print("\nKiểu dữ liệu:")
    print(df.dtypes)
    print("\nThống kê mô tả:")
    print(df.describe())
    print("\nSố lượng giá trị null:")
    print(df.isnull().sum())

    # Vẽ biểu đồ phân phối giá gốc
    plt.figure(figsize=(10, 6))
    sns.histplot(df[ORIGINAL_TARGET], kde=True)
    plt.title('Phân phối giá nhà (Gốc)')
    plt.savefig('price_distribution_original.png')
    plt.close()

    # Vẽ ma trận tương quan
    plt.figure(figsize=(14, 12)) # Tăng kích thước
    numeric_cols = df.select_dtypes(include=np.number).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5) # Điều chỉnh fmt
    plt.title('Ma trận tương quan')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def handle_outliers(df, column):
    """Xử lý outliers cho cột chỉ định bằng phương pháp IQR."""
    print(f"\n4. XỬ LÝ OUTLIERS cho cột {column}")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    print(f"Số lượng outliers tìm thấy: {outliers_count}")

    df[TARGET_COLUMN] = df[column].clip(lower_bound, upper_bound)
    print(f"Đã tạo cột {TARGET_COLUMN} đã xử lý outliers.")

    # Vẽ boxplot so sánh
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column])
    plt.title(f'{column} (Trước xử lý)')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[TARGET_COLUMN])
    plt.title(f'{TARGET_COLUMN} (Sau xử lý)')
    plt.tight_layout()
    plt.savefig('outliers_handling.png')
    plt.close()
    return df

def feature_engineering(df):
    """Tạo các biến mới từ dữ liệu hiện có."""
    print("\n6. TẠO BIẾN MỚI (Feature Engineering)")
    # Xử lý chia cho 0 hoặc null
    df['PRICE_PER_SQFT'] = (df[ORIGINAL_TARGET] / df['LND_SQFOOT'].replace(0, np.nan)).fillna(0)
    df['LIVING_LAND_RATIO'] = (df['TOT_LVG_AREA'] / df['LND_SQFOOT'].replace(0, np.nan)).fillna(0)
    df['AVG_IMPORTANT_DIST'] = df[['OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'HWY_DIST']].mean(axis=1)
    print("Đã tạo các biến: PRICE_PER_SQFT, LIVING_LAND_RATIO, AVG_IMPORTANT_DIST.")
    return df

def create_preprocessor(numeric_features, categorical_features):
    """Tạo bộ tiền xử lý ColumnTransformer."""
    print("\n9. TẠO BỘ TIỀN XỬ LÝ")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Giữ lại các cột không được xử lý nếu có
    )
    print("Bộ tiền xử lý đã được tạo.")
    return preprocessor

def compare_base_models(X_train, y_train, preprocessor, models):
    """So sánh hiệu suất các mô hình cơ bản bằng cross-validation."""
    print("\n10. SO SÁNH CÁC MÔ HÌNH CƠ BẢN")
    results = {}
    for name, model in models.items():
        start_time = time.time()
        # Tạo pipeline hoàn chỉnh cho từng mô hình
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # Đánh giá trên pipeline hoàn chỉnh
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=kfold, scoring='neg_mean_squared_error', n_jobs=N_JOBS)
        rmse_scores = np.sqrt(-cv_scores)
        results[name] = rmse_scores
        elapsed_time = time.time() - start_time
        print(f"{name}: RMSE = {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f}) - Time: {elapsed_time:.2f}s")

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.boxplot(results.values(), labels=results.keys())
    plt.title('So sánh hiệu suất RMSE các mô hình (Cross-Validation)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    # Xác định mô hình tốt nhất dựa trên RMSE trung bình
    best_model_name = min(results, key=lambda name: np.mean(results[name]))
    print(f"\nMô hình cơ bản tốt nhất (dựa trên CV RMSE): {best_model_name}")
    return best_model_name


def tune_hyperparameters(X_train, y_train, preprocessor, base_model, param_grid):
    """Tinh chỉnh siêu tham số cho mô hình tốt nhất bằng GridSearchCV."""
    print(f"\n11. TINH CHỈNH SIÊU THAM SỐ cho {type(base_model).__name__}")
    
    # Tạo pipeline hoàn chỉnh để tinh chỉnh
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', base_model)])

    print("Bắt đầu Grid Search...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=GRID_SEARCH_CV_FOLDS,
                               scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=1)
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Grid Search hoàn thành trong {elapsed_time:.2f}s")

    print(f"\nBộ tham số tốt nhất tìm được:")
    # Loại bỏ tiền tố 'model__' khỏi tên tham số để dễ đọc hơn
    best_params_cleaned = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    print(best_params_cleaned)
    print(f"RMSE tốt nhất trên tập huấn luyện (CV): {np.sqrt(-grid_search.best_score_):.2f}")

    return grid_search.best_estimator_ # Trả về pipeline tốt nhất

def plot_learning_curve(estimator, X, y, title):
    """Vẽ learning curve cho mô hình."""
    print(f"\n13. VẼ LEARNING CURVE cho {title}")
    start_time = time.time()
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=CV_FOLDS, scoring='neg_mean_squared_error', n_jobs=N_JOBS,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=RANDOM_STATE
    )
    elapsed_time = time.time() - start_time
    print(f"Vẽ learning curve hoàn thành trong {elapsed_time:.2f}s")

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    train_rmse_std = np.sqrt(train_scores.std(axis=1)) # Std dev của score, không cần lấy căn
    val_rmse_std = np.sqrt(val_scores.std(axis=1))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, 'o-', label='Training RMSE')
    plt.plot(train_sizes, val_rmse, 'o-', label='Validation RMSE')
    plt.fill_between(train_sizes, train_rmse - train_rmse_std, train_rmse + train_rmse_std, alpha=0.1)
    plt.fill_between(train_sizes, val_rmse - val_rmse_std, val_rmse + val_rmse_std, alpha=0.1)
    plt.xlabel('Kích thước tập huấn luyện')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curve: {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """Đánh giá mô hình cuối cùng trên tập test."""
    print(f"\n14. ĐÁNH GIÁ MÔ HÌNH '{model_name}' TRÊN TẬP TEST")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # Xử lý trường hợp y_test = 0 để tránh lỗi chia cho 0
    y_test_safe = np.where(y_test == 0, 1e-6, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100

    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Vẽ biểu đồ dự đoán vs thực tế
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title(f'Giá thực tế vs. Giá dự đoán ({model_name})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png')
    plt.close()

    # Vẽ biểu đồ phân phối sai số
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=50)
    plt.xlabel('Sai số dự đoán (Thực tế - Dự đoán)')
    plt.ylabel('Số lượng')
    plt.title(f'Phân phối sai số dự đoán ({model_name})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

def feature_importance_analysis(model, X_train):
    """Phân tích độ quan trọng của đặc trưng (nếu mô hình hỗ trợ)."""
    print("\n15. PHÂN TÍCH ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG")
    
    # Truy cập mô hình bên trong pipeline
    try:
        # Thử truy cập trực tiếp nếu model không phải là pipeline (trường hợp hiếm)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            model_step = model # Không có bước tiền xử lý
        # Truy cập mô hình trong pipeline
        elif hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'feature_importances_'):
             model_step = model.named_steps['model']
             importances = model_step.feature_importances_
        else:
             print("Mô hình này không hỗ trợ phân tích độ quan trọng.")
             return

        # Lấy tên đặc trưng sau khi tiền xử lý (phức tạp hơn với OneHotEncoder)
        # Cách đơn giản: Lấy tên gốc từ X_train (chỉ đúng nếu không có OneHotEncoder hoặc giữ nguyên thứ tự)
        # Lưu ý: Cách này không hoàn toàn chính xác nếu có OneHotEncoder vì nó tạo nhiều cột mới
        # Để chính xác hơn cần sử dụng get_feature_names_out() từ preprocessor
        
        # Thử lấy tên đặc trưng từ preprocessor (nếu có thể)
        feature_names_out = []
        try:
            preprocessor_step = model.named_steps.get('preprocessor')
            if preprocessor_step:
                 feature_names_out = preprocessor_step.get_feature_names_out()
            else: # Nếu không có preprocessor trong pipeline
                 feature_names_out = X_train.columns.tolist()
        except Exception as e:
            print(f"Không thể lấy tên đặc trưng tự động từ preprocessor: {e}. Sử dụng tên cột gốc.")
            feature_names_out = X_train.columns.tolist() # Fallback

        # Đảm bảo số lượng tên đặc trưng khớp với số lượng importances
        if len(feature_names_out) != len(importances):
             print(f"Cảnh báo: Số lượng tên đặc trưng ({len(feature_names_out)}) không khớp với số lượng importances ({len(importances)}). Sử dụng tên cột gốc.")
             feature_names_out = X_train.columns.tolist()
             # Kiểm tra lại nếu vẫn không khớp
             if len(feature_names_out) != len(importances):
                  print("Vẫn không khớp. Bỏ qua vẽ biểu đồ.")
                  return


        feature_importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        print("\nTop 15 đặc trưng quan trọng nhất:")
        print(feature_importance_df.head(15))

        # Vẽ biểu đồ
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
        plt.title('Độ quan trọng của 15 đặc trưng hàng đầu')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    except Exception as e:
        print(f"Lỗi khi phân tích độ quan trọng: {e}")


# --- Quy trình chính ---
if __name__ == "__main__":
    print("=== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN GIÁ NHÀ ===")
    start_total_time = time.time()

    # 1 & 2. Đọc và Khám phá dữ liệu
    df = load_data(DATA_FILE)
    explore_data(df)

    # 3. Xử lý dữ liệu thiếu (được xử lý trong pipeline bằng SimpleImputer)
    print("\n3. XỬ LÝ DỮ LIỆU THIẾU (sẽ thực hiện trong pipeline)")

    # 4. Xử lý Outliers cho biến mục tiêu
    df = handle_outliers(df, ORIGINAL_TARGET)

    # 5. Mã hóa biến phân loại (được xử lý trong pipeline bằng OneHotEncoder)
    print("\n5. MÃ HÓA BIẾN PHÂN LOẠI (sẽ thực hiện trong pipeline)")

    # 6. Tạo biến mới
    df = feature_engineering(df)

    # 7. Chuẩn bị dữ liệu X, y
    print("\n7. CHUẨN BỊ DỮ LIỆU X, y")
    features_to_drop = [ORIGINAL_TARGET, TARGET_COLUMN, ID_COLUMN]
    # Tự động xác định kiểu dữ liệu để tránh lỗi nếu cột không tồn tại
    features_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=features_to_drop)
    y = df[TARGET_COLUMN]
    print(f"Các đặc trưng (X shape): {X.shape}")
    print(f"Biến mục tiêu (y shape): {y.shape}")

    # Xác định các cột số và phân loại
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


    # 8. Chia dữ liệu
    print("\n8. CHIA DỮ LIỆU")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Kích thước tập huấn luyện: X={X_train.shape}, y={y_train.shape}")
    print(f"Kích thước tập kiểm tra: X={X_test.shape}, y={y_test.shape}")

    # 9. Tạo bộ tiền xử lý
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    # 10. So sánh các mô hình cơ bản
    best_base_model_name = compare_base_models(X_train, y_train, preprocessor, BASE_MODELS)
    # Chọn mô hình Gradient Boosting để tinh chỉnh (hoặc dùng best_base_model_name)
    model_to_tune_name = 'Gradient Boosting'
    model_to_tune = BASE_MODELS[model_to_tune_name]

    # 11. Tinh chỉnh siêu tham số cho mô hình tốt nhất
    # Chỉ tinh chỉnh nếu mô hình là Gradient Boosting
    if isinstance(model_to_tune, GradientBoostingRegressor):
         best_pipeline = tune_hyperparameters(X_train, y_train, preprocessor, model_to_tune, GB_PARAM_GRID)
    else:
         print(f"\nBỏ qua tinh chỉnh siêu tham số cho {model_to_tune_name}.")
         # Tạo pipeline cuối cùng với mô hình cơ bản tốt nhất không cần tinh chỉnh
         best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('model', BASE_MODELS[best_base_model_name])])
         best_pipeline.fit(X_train, y_train) # Huấn luyện pipeline cuối cùng này


    # 12. Luyện mô hình tốt nhất (đã thực hiện trong GridSearchCV hoặc fit ở trên)
    print(f"\n12. MÔ HÌNH TỐT NHẤT ({type(best_pipeline.named_steps['model']).__name__}) ĐÃ ĐƯỢC HUẤN LUYỆN")


    # 13. Vẽ Learning Curve
    plot_learning_curve(best_pipeline, X_train, y_train, type(best_pipeline.named_steps['model']).__name__)

    # 14. Đánh giá mô hình trên tập test
    evaluate_model(best_pipeline, X_test, y_test, type(best_pipeline.named_steps['model']).__name__)

    # 15. Phân tích độ quan trọng của đặc trưng
    # Cần truyền X_train gốc để lấy tên cột đúng
    feature_importance_analysis(best_pipeline, X_train)

    # Lưu mô hình và bộ tiền xử lý (lưu toàn bộ pipeline)
    joblib.dump(best_pipeline, 'best_pipeline_optimized.pkl')
    # Không cần lưu preprocessor riêng vì nó đã nằm trong pipeline
    print("\nPipeline tốt nhất đã được lưu vào file 'best_pipeline_optimized.pkl'")

    end_total_time = time.time()
    print(f"\n=== QUY TRÌNH HOÀN THÀNH TRONG {end_total_time - start_total_time:.2f}s ===")

    # Giả sử bạn đã có các giá trị RMSE cho cả hai mô hình
    rmse_random_forest = 14779.87
    rmse_gradient_boosting = 16168.27

    models = ['Random Forest', 'Gradient Boosting']
    rmse_values = [rmse_random_forest, rmse_gradient_boosting]

    plt.figure(figsize=(8, 6))
    plt.bar(models, rmse_values, color=['blue', 'green'])
    plt.ylabel('RMSE')
    plt.title('So sánh RMSE giữa Random Forest và Gradient Boosting')
    plt.show() 