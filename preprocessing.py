import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from config import *
from data_loader import load_data

logger = logging.getLogger(__name__)

def handle_outliers(df, column, print_info=True):
    """Xử lý outliers cho cột chỉ định bằng phương pháp IQR."""
    logger.info(f"Xử lý outliers cho cột {column}")
    
    if column not in df.columns:
        raise ValueError(f"Cột {column} không tồn tại trong dữ liệu")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    if print_info:
        print(f"Số lượng outlier tìm thấy: {outliers_count}")
        print(f"Đã tạo cột {TARGET_COLUMN} đã xử lý outliers.")
    logger.info(f"Số outliers: {outliers_count}")

    df[TARGET_COLUMN] = df[column].clip(lower_bound, upper_bound)
    logger.info(f"Đã tạo cột {TARGET_COLUMN}")

    # Vẽ boxplot trước/sau xử lý trên cùng 1 ảnh
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df[column])
        plt.title(f'{column} (Trước xử lý)')
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[TARGET_COLUMN])
        plt.title(f'{TARGET_COLUMN} (Sau xử lý)')
        plt.tight_layout()
        save_path = os.path.join(PREPROCESSING_VIS_DIR, 'boxplot_sale_prc_compare.png')
        plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
        plt.close()
        if print_info:
            print(f"Đã lưu boxplot so sánh trước/sau xử lý outlier tại: {save_path}")
    except Exception as e:
        logger.error(f"Lỗi khi vẽ boxplot so sánh: {str(e)}")
    return df

def feature_engineering(df):
    """Tạo các biến mới từ dữ liệu hiện có (chỉ tạo nếu đủ cột)."""
    logger.info("Bắt đầu tạo biến mới (Feature Engineering)")
    created_features = []

    # 1. Giá trên mỗi feet vuông đất
    if 'LND_SQFOOT' in df.columns and 'SALE_PRC' in df.columns:
        df['PRICE_PER_SQFT'] = (df['SALE_PRC'] / df['LND_SQFOOT'].replace(0, np.nan)).fillna(0)
        created_features.append('PRICE_PER_SQFT')

    # 2. Tỷ lệ diện tích sống trên diện tích đất
    if 'TOT_LVG_AREA' in df.columns and 'LND_SQFOOT' in df.columns:
        df['LIVING_LAND_RATIO'] = (df['TOT_LVG_AREA'] / df['LND_SQFOOT'].replace(0, np.nan)).fillna(0)
        created_features.append('LIVING_LAND_RATIO')

    # 3. Khoảng cách trung bình đến các điểm quan trọng
    dist_cols = [c for c in ['OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'HWY_DIST'] if c in df.columns]
    if len(dist_cols) > 0:
        df['AVG_IMPORTANT_DIST'] = df[dist_cols].mean(axis=1)
        created_features.append('AVG_IMPORTANT_DIST')

    # Vẽ phân phối của các biến mới (nếu muốn)
    for feature in created_features:
        try:
            plt.figure(figsize=FIGURE_SIZE)
            sns.histplot(df[feature], kde=True)
            plt.title(f'Phân phối của {feature}')
            plt.savefig(os.path.join(PREPROCESSING_VIS_DIR, f'numeric_{feature}_transformation.png'), 
                        dpi=DPI, format=SAVE_FORMAT)
            plt.close()
        except Exception as e:
            logger.error(f"Lỗi khi vẽ biểu đồ cho {feature}: {str(e)}")

    logger.info(f"Đã tạo {len(created_features)} biến mới: {', '.join(created_features)}")
    return df, created_features

def create_preprocessor(numeric_features, categorical_features):
    """Tạo bộ tiền xử lý ColumnTransformer."""
    logger.info("Tạo bộ tiền xử lý")
    
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
        remainder='passthrough'
    )
    
    logger.info("Đã tạo bộ tiền xử lý")
    return preprocessor

def prepare_data_for_model(df, target_column=TARGET_COLUMN):
    """Chuẩn bị dữ liệu cho model."""
    logger.info("Chuẩn bị dữ liệu cho model")
    
    X = df.drop(columns=[target_column, ORIGINAL_TARGET])
    y = df[target_column]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    logger.info(f"Features: {len(numeric_features)} biến số, {len(categorical_features)} biến phân loại")
    
    return X, y, numeric_features, categorical_features

def transform_data(X_train, X_test, preprocessor):
    """Áp dụng preprocessor cho dữ liệu train và test."""
    logger.info("Transform dữ liệu")
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    logger.info(f"Train shape: {X_train_transformed.shape}, Test shape: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed 

def run_preprocessing_steps(df, verbose=False):
    """
    Thực hiện các bước tiền xử lý dữ liệu (xử lý giá trị thiếu, outliers, tạo biến mới, chuẩn hóa, chia train/test) và trả về các biến cần thiết cho các bước tiếp theo.
    Trả về: (df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, preprocessor).
    """
    print("\n=== BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ===")
    print("\n1. XỬ LÝ GIÁ TRỊ THIẾU (NaN) bằng SimpleImputer (mean cho biến số, most_frequent cho biến phân loại) trong bộ tiền xử lý.")
    print("\n4. XỬ LÝ OUTLIERS cho cột SALE_PRC")
    df = handle_outliers(df, 'SALE_PRC', print_info=True)
    print("\n6. TẠO BIẾN MỚI (Feature Engineering)")
    df, created_features = feature_engineering(df)
    if created_features:
        print("Đã tạo các biến mới:", ", ".join(created_features))
    else:
        print("Không có biến mới nào được tạo.")
    if 'PARCELNO' in df.columns:
        df.drop(columns=['PARCELNO'], inplace=True)
        print("Đã xóa cột PARCELNO khỏi dữ liệu.")
    # Ép kiểu các biến phân loại dạng số
    for col in ['month_sold', 'structure_quality', 'avno60plus']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    # Chuẩn bị dữ liệu cho model
    from sklearn.model_selection import train_test_split
    X, y, numeric_features, categorical_features = prepare_data_for_model(df)
    print("Các biến số:", list(numeric_features))
    print("Các biến phân loại:", list(categorical_features))
    # Tạo preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    if len(categorical_features) > 0:
        print(f"\nĐã mã hóa các biến phân loại (OneHotEncoder): {', '.join(categorical_features)}.")
    else:
        print("\nKhông có biến phân loại nào để mã hóa OneHotEncoder.")
    print("Đã chuẩn hóa các biến số bằng StandardScaler.")
    print(f"\nĐã tạo bộ tiền xử lý với {len(numeric_features)} biến số, {len(categorical_features)} biến phân loại.")
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"\nKích thước tập huấn luyện: X={X_train.shape}, y={y_train.shape}")
    print(f"Kích thước tập kiểm tra: X={X_test.shape}, y={y_test.shape}")

    print("\nĐã hoàn thành preprocessing.")
    return (df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, preprocessor)

def log_transform_data(X_train, X_test, y_train, y_test):
    """
    Thực hiện log transform cho biến mục tiêu và các biến độc lập có |skewness| > 1.0.
    In bảng skewness trước/sau, vẽ biểu đồ so sánh cho biến mục tiêu và 3 biến quan trọng.
    Đảm bảo không NaN, nhất quán train/test.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    # 1. Chuyển đổi biến mục tiêu
    print("\n=== BƯỚC 1.4: CHUYỂN ĐỔI DỮ LIỆU (LOG TRANSFORM) ===")
    # Skewness trước
    skew_before = y_train.skew()
    # Vẽ so sánh trước/sau + boxplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(y_train, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Trước log transform')
    y_train_log = np.log1p(y_train)
    sns.histplot(y_train_log, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Sau log transform')
    # Boxplot long-form
    df_box = pd.DataFrame({
        'Giá trị': pd.concat([y_train, y_train_log], ignore_index=True),
        'Trạng thái': ['Trước'] * len(y_train) + ['Sau'] * len(y_train_log)
    })
    sns.boxplot(x='Trạng thái', y='Giá trị', data=df_box, ax=axes[2])
    axes[2].set_title('Boxplot so sánh')
    plt.tight_layout()
    save_path = os.path.join(PREPROCESSING_VIS_DIR, 'target_log_transform_compare.png')
    plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh biến mục tiêu trước/sau log transform tại: {save_path}")
    # Skewness sau
    skew_after = y_train_log.skew()
    print(f"Độ lệch ban đầu của biến mục tiêu: {skew_before:.3f}, sau log transform: {skew_after:.3f}, cải thiện: {abs(skew_before) - abs(skew_after):.3f}")
    # Gán lại biến mục tiêu
    y_train = y_train_log
    y_test = np.log1p(y_test)

    # 2. Chuyển đổi các biến độc lập có |skewness| > 1.0 (chỉ áp dụng cho biến số)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    skewness_train = X_train[numeric_cols].skew().sort_values(ascending=False)
    skewed_cols = skewness_train[abs(skewness_train) > 1.0].index.tolist()
    print(f"\nCác biến độc lập có |skewness| > 1.0: {skewed_cols}")
    # Lưu bảng skewness trước/sau và phép biến đổi
    skew_table = pd.DataFrame({'Trước': X_train[skewed_cols].skew()})
    transform_notes = {}
    for col in skewed_cols:
        min_val = X_train[col].min()
        if min_val < 0:
            X_train[col] = np.log1p(X_train[col] - min_val + 1)
            X_test[col] = np.log1p(X_test[col] - min_val + 1)
            transform_notes[col] = f'log1p(x - {min_val:.2f} + 1)'
        else:
            X_train[col] = np.log1p(X_train[col])
            X_test[col] = np.log1p(X_test[col])
            transform_notes[col] = 'log1p'
    skew_table['Sau'] = X_train[skewed_cols].skew()
    skew_table['Cải thiện'] = (abs(skew_table['Trước']) - abs(skew_table['Sau'])).round(3)
    skew_table['Phép biến đổi'] = [transform_notes[col] for col in skewed_cols]
    print("\n=== TÓM TẮT KẾT QUẢ CHUYỂN ĐỔI ===")
    print(f"Số biến đã chuyển đổi: {len(skewed_cols)}/{X_train.shape[1]}")
    print("\nChi tiết các biến đã chuyển đổi:")
    print(skew_table)

    # Vẽ biểu đồ so sánh cho 3 biến quan trọng
    for col in ['LND_SQFOOT', 'TOT_LVG_AREA', 'WATER_DIST']:
        if col in X_train.columns:
            # Lấy lại giá trị trước log bằng expm1 (vì X_train[col] đã log1p)
            before = np.expm1(X_train[col])
            after = X_train[col]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(before, kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f'{col} trước log')
            sns.histplot(after, kde=True, ax=axes[1], color='orange')
            axes[1].set_title(f'{col} sau log')
            # Boxplot long-form
            df_box = pd.DataFrame({
                'Giá trị': pd.concat([before, after], ignore_index=True),
                'Trạng thái': ['Trước'] * len(before) + ['Sau'] * len(after)
            })
            sns.boxplot(x='Trạng thái', y='Giá trị', data=df_box, ax=axes[2])
            axes[2].set_title('Boxplot so sánh')
            plt.tight_layout()
            save_path = os.path.join(PREPROCESSING_VIS_DIR, f'{col}_log_transform_compare.png')
            plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
            plt.close()
            print(f"Đã lưu biểu đồ so sánh log transform cho {col} tại: {save_path}")

    # Kiểm tra NaN
    nan_train = X_train.isna().sum().sum() + y_train.isna().sum()
    nan_test = X_test.isna().sum().sum() + y_test.isna().sum()
    print(f"\nKiểm tra NaN sau chuyển đổi: train={nan_train}, test={nan_test}")
    if nan_train == 0 and nan_test == 0:
        print("Dữ liệu sau chuyển đổi không chứa giá trị NaN.")
    else:
        print("Cảnh báo: Dữ liệu sau chuyển đổi còn NaN!")

    print(f"\nĐã chuyển đổi biến mục tiêu và {len(skewed_cols)}/{X_train.shape[1]} biến độc lập.")
    print("Dữ liệu đã sẵn sàng cho việc huấn luyện mô hình.")
    return X_train, X_test, y_train, y_test

def analyze_training_data(X_train, y_train, numeric_features, categorical_features):
    """
    Thực hiện phân tích thống kê dữ liệu mẫu (tập huấn luyện) theo yêu cầu:
    – In thống kê mô tả (y_train.describe) và vẽ histogram cho biến mục tiêu.
    – In thống kê mô tả (X_train[numeric_features].describe) và vẽ ma trận tương quan cho các biến số.
    – Phân tích các biến phân loại (nếu có) (in thông báo nếu không có, vẽ phân phối cho từng biến phân loại).
    Output được tách biệt (in ra mục riêng) để đồng bộ với hàm main.
    """
    print("\n=== BƯỚC 1.3: THỐNG KÊ DỮ LIỆU MẪU (TẬP HUẤN LUYỆN) ===")
    # 1. Phân tích biến mục tiêu (y_train)
    print("\n1. Phân tích biến mục tiêu (y_train):")
    print(y_train.describe())
    plt.figure(figsize=(FIGURE_SIZE))
    sns.histplot(y_train, kde=True, bins=80, color='steelblue', edgecolor=None)
    plt.title("Phân phối biến mục tiêu (y_train)")
    plt.xlabel("y_train")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(PREPROCESSING_VIS_DIR, "y_train_distribution.png")
    plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    print("Đã lưu biểu đồ phân phối biến mục tiêu tại: " + save_path)

    # 2. Phân tích biến số (X_train[numeric_features])
    print("\n2. Phân tích biến số (X_train[numeric_features]):")
    print(X_train[numeric_features].describe())
    plt.figure(figsize=(FIGURE_SIZE))
    corr = X_train[numeric_features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
    plt.title("Ma trận tương quan các biến số (X_train)")
    plt.tight_layout()
    save_path = os.path.join(PREPROCESSING_VIS_DIR, "correlation_matrix_train.png")
    plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    print("Đã lưu ma trận tương quan tại: " + save_path)

    # 3. Phân tích biến phân loại (nếu có)
    print("\n3. Phân tích biến phân loại (X_train[categorical_features]):")
    if len(categorical_features) == 0:
        print("Không có biến phân loại nào trong tập huấn luyện.")
    else:
        print("Danh sách các biến phân loại (có ý nghĩa phân loại): " + ", ".join(categorical_features))
        for col in categorical_features:
            plt.figure(figsize=FIGURE_SIZE)
            if col == "structure_quality":
                order = [str(i) for i in range(6)]  # 0-5
                sns.countplot(x=X_train[col].astype(str), order=order)
            elif col == "month_sold":
                order = [str(i) for i in range(1, 13)]  # 1-12
                sns.countplot(x=X_train[col].astype(str), order=order)
            else:
                sns.countplot(x=X_train[col].astype(str))
            plt.title(f"Phân phối '{col}' (X_train) - ĐÃ CHUẨN HOÁ TRỤC OX")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=0)  # chữ đứng
            plt.tight_layout()
            save_path = os.path.join(PREPROCESSING_VIS_DIR, f"categorical_{col}_train.png")
            plt.savefig(save_path, dpi=DPI, format=SAVE_FORMAT)
            plt.close()
            print(f"[OK] Đã lưu biểu đồ phân phối cho biến phân loại {col} tại: {save_path}")
    print("\nĐã hoàn thành phân tích thống kê dữ liệu mẫu (tập huấn luyện).")

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    # Chạy preprocessing và lấy các biến cần thiết
    df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, preprocessor = run_preprocessing_steps(df, verbose=True)
    # 1.3: Phân tích dữ liệu mẫu (trước chuyển đổi)
    analyze_training_data(X_train, y_train, numeric_features, categorical_features)
    # 1.4: Chuyển đổi dữ liệu (log transform)
    X_train, X_test, y_train, y_test = log_transform_data(X_train, X_test, y_train, y_test) 