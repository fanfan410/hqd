import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from data_loader import load_data, explore_data, analyze_missing_values, analyze_target_distribution
from preprocessing import (
    handle_outliers, feature_engineering, create_preprocessor,
    prepare_data_for_model, transform_data, run_preprocessing_steps, analyze_training_data, log_transform_data
)
from model_training import (
    compare_base_models, tune_hyperparameters,
    train_final_model, plot_learning_curve
)
from model_evaluation import generate_evaluation_report
from visualization import create_visualization_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor

# Tắt toàn bộ cảnh báo sklearn
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Thiết lập logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Chạy toàn bộ pipeline từ đầu đến cuối."""
    try:
        # 1. Load và khám phá dữ liệu
        print("\n=== BƯỚC 1: LOAD VÀ KHÁM PHÁ DỮ LIỆU ===")
        logger.info("Bắt đầu pipeline")
        df = load_data()
        explore_data(df)
        analyze_missing_values(df)
        analyze_target_distribution(df, target_column='SALE_PRC')

        # 2. Tiền xử lý dữ liệu (đồng bộ output, chỉ gọi hàm tổng hợp)
        print("\n=== BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ===")
        logger.info("Bắt đầu tiền xử lý dữ liệu")
        df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, preprocessor = run_preprocessing_steps(df)
        # 1.3: Phân tích dữ liệu mẫu (trước chuyển đổi)
        analyze_training_data(X_train, y_train, numeric_features, categorical_features)
        # 1.4: Chuyển đổi dữ liệu (log transform)
        X_train, X_test, y_train, y_test = log_transform_data(X_train, X_test, y_train, y_test)
        
        # 3. Tạo báo cáo trực quan hóa
        print("\n=== BƯỚC 3: TẠO BÁO CÁO TRỰC QUAN HÓA ===")
        logger.info("Tạo báo cáo trực quan hóa")
        create_visualization_report(
            df, list(numeric_features), list(categorical_features), TARGET_COLUMN
        )
        print("✓ Hoàn thành tạo báo cáo trực quan hóa")
        
        # 4. So sánh các mô hình cơ bản
        print("\n=== BƯỚC 4: SO SÁNH CÁC MÔ HÌNH CƠ BẢN ===")
        print("Đang chạy cross-validation cho các model... (có thể mất vài phút)")
        logger.info("So sánh các mô hình cơ bản")
        model_comparison_results = compare_base_models(X_train, y_train, preprocessor)
        print("✓ Hoàn thành so sánh các mô hình cơ bản")
        
        # 5. Tinh chỉnh hyperparameters cho mô hình tốt nhất
        print("\n=== BƯỚC 5: TINH CHỈNH HYPERPARAMETERS ===")
        print("Đang chạy GridSearchCV... (có thể mất vài phút)")
        logger.info("Tinh chỉnh hyperparameters")
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5]
        }
        best_model, best_params = tune_hyperparameters(
            X_train, y_train,
            preprocessor, GradientBoostingRegressor(),
            param_grid
        )
        print("✓ Hoàn thành tinh chỉnh hyperparameters")
        
        # 6. Huấn luyện mô hình cuối cùng
        print("\n=== BƯỚC 6: HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG ===")
        logger.info("Huấn luyện mô hình cuối cùng")
        final_model = train_final_model(
            X_train, y_train,
            preprocessor, best_params
        )
        print("✓ Hoàn thành huấn luyện mô hình cuối cùng")
        
        # 7. Vẽ learning curve
        print("\n=== BƯỚC 7: VẼ LEARNING CURVE ===")
        logger.info("Vẽ learning curve")
        plot_learning_curve(
            final_model, X_train, y_train,
            "Learning Curve của Mô hình Cuối cùng"
        )
        print("✓ Hoàn thành vẽ learning curve")
        
        # 8. Đánh giá mô hình
        print("\n=== BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH ===")
        logger.info("Đánh giá mô hình")
        evaluation_report = generate_evaluation_report(
            final_model, X_test, y_test,
            list(numeric_features) + list(categorical_features)
        )
        print("✓ Hoàn thành đánh giá mô hình")
        
        print("\nThống kê biến mục tiêu (y):")
        print(y.describe())
        
        print("\nThống kê các biến số:")
        print(X.describe())
        
        print("Danh sách các biến phân loại:")
        print(list(categorical_features))
        if not list(categorical_features):
            print("\nKhông có biến phân loại nào trong dataset")
            print("Tất cả các biến đều là biến số (numeric)")
        
        print("\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE! 🎉")
        logger.info("Hoàn thành pipeline")
        
    except Exception as e:
        logger.error(f"Lỗi trong pipeline: {str(e)}", exc_info=True)
        raise

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

if __name__ == "__main__":
    main() 