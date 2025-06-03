import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from config import *
import warnings
from sklearn.exceptions import ConvergenceWarning
import scipy.stats as stats
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from category_encoders import CatBoostEncoder

# Định nghĩa các thư mục
MODEL_DIR = os.path.join('models', 'trained')
MODEL_TRAINING_VIS_DIR = os.path.join('visualizations', 'model_training')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_TRAINING_VIS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Danh sách các mô hình cần thử nghiệm
BASE_MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0, max_iter=20000),
    'Elastic Net': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000, tol=1e-4),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method='hist',
        device = 'cuda',
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0
    ),
    'Stacking Ensemble': StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
            ('xgb', XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=N_JOBS, tree_method='hist', device='cuda')),
            ('ridge', Ridge(alpha=1.0, max_iter=10000, random_state=RANDOM_STATE))
        ],
        final_estimator=Ridge(alpha=1.0, max_iter=10000, random_state=RANDOM_STATE),
        n_jobs=N_JOBS,
        passthrough=False
    )
}

def preprocess_features(X_train, X_test, numeric_features, categorical_features):
    """
    Tiền xử lý dữ liệu trước khi huấn luyện mô hình.
    Sử dụng RobustScaler cho biến số để xử lý outliers tốt hơn.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('catboost', CatBoostEncoder())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0
    )
    
    return preprocessor

def compare_base_models(X_train, y_train, X_test, y_test, preprocessor, models=None):
    """
    So sánh các mô hình cơ bản và trả về mô hình tốt nhất.
    """
    if models is None:
        models = BASE_MODELS
    
    results = {}
    
    for name, model in models.items():
        print(f"\nĐang huấn luyện {name}...")
        
        try:
            # Tạo pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Đo thời gian huấn luyện
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Đánh giá trên tập test
            y_pred = pipeline.predict(X_test)
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=CV_FOLDS,
                scoring='r2',
                n_jobs=N_JOBS
            )
            
            # Lưu kết quả
            results[name] = {
                'pipeline': pipeline,
                'metrics': metrics,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std()
                },
                'training_time': training_time
            }
            
            print(f"Hoàn thành {name} - R²: {metrics['R2']:.3f}")
            
        except Exception as e:
            print(f"Lỗi khi huấn luyện {name}: {str(e)}")
            continue
    
    if not results:
        raise Exception("Không có mô hình nào được huấn luyện thành công")
    
    # Vẽ biểu đồ so sánh
    plot_model_comparison(results)
    
    # Chọn mô hình tốt nhất dựa trên R2
    best_model_name = max(results.items(), 
                         key=lambda x: x[1]['metrics']['R2'])[0]
    
    # Chỉ vẽ learning curve cho mô hình tốt nhất
    print(f"\nPhân tích learning curve cho mô hình tốt nhất ({best_model_name})...")
    learning_analysis = plot_learning_curve(
        results[best_model_name]['pipeline'], 
        X_train, y_train, 
        f"Learning Curve - {best_model_name}"
    )
    
    # In thông tin phân tích learning curve
    print("\nPhân tích learning curve:")
    print(f"- Khoảng cách lớn nhất giữa train và test: {learning_analysis['gap_analysis']['max_gap']:.3f}")
    print(f"- Khoảng cách trung bình: {learning_analysis['gap_analysis']['avg_gap']:.3f}")
    print(f"- Tỷ lệ tốc độ học (train/test): {learning_analysis['learning_rate']['rate_ratio']:.2f}")
    
    return best_model_name, results

def tune_hyperparameters(X_train, y_train, preprocessor, base_model, param_grid):
    """
    Tinh chỉnh siêu tham số cho mô hình.
    
    Args:
        X_train, y_train: Dữ liệu huấn luyện
        preprocessor: Pipeline tiền xử lý
        base_model: Mô hình cơ bản cần tinh chỉnh
        param_grid: Dict chứa các tham số cần tinh chỉnh
        
    Returns:
        Pipeline tốt nhất sau khi tinh chỉnh
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring='r2',
        n_jobs=N_JOBS,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"\nKết quả tinh chỉnh tham số:")
    logger.info(f"Tham số tốt nhất: {grid_search.best_params_}")
    logger.info(f"Điểm R2 tốt nhất: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def plot_learning_curve(estimator, X, y, title, cv=3, n_jobs=-1):
    """
    Vẽ learning curve chi tiết cho mô hình tốt nhất.
    """
    # Tính toán learning curve với ít điểm hơn
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring='r2',
        random_state=RANDOM_STATE
    )
    
    # Tính toán mean và std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Learning curve cơ bản
    ax1.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    ax1.plot(train_sizes, test_mean, label='Cross-validation score', color='red', marker='o')
    
    # Vẽ vùng std
    ax1.fill_between(train_sizes, 
                     train_mean - train_std,
                     train_mean + train_std, 
                     alpha=0.1, color='blue')
    ax1.fill_between(train_sizes, 
                     test_mean - test_std,
                     test_mean + test_std, 
                     alpha=0.1, color='red')
    
    ax1.set_xlabel('Số Lượng Mẫu Huấn Luyện')
    ax1.set_ylabel('Điểm R²')
    ax1.set_title('Learning Curve')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # Thêm thông tin về gap
    gap = train_mean - test_mean
    ax1.text(0.05, 0.05, 
             f'Khoảng Cách Lớn Nhất: {np.max(gap):.3f}\n'
             f'Khoảng Cách Nhỏ Nhất: {np.min(gap):.3f}',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Phân tích chi tiết
    # Tính toán tốc độ học
    train_slopes = np.gradient(train_mean)
    test_slopes = np.gradient(test_mean)
    
    # Vẽ tốc độ học
    ax2.plot(train_sizes[1:], train_slopes[1:], 
             label='Tốc Độ Học (Training)', color='blue', marker='o')
    ax2.plot(train_sizes[1:], test_slopes[1:], 
             label='Tốc Độ Học (CV)', color='red', marker='o')
    
    ax2.set_xlabel('Số Lượng Mẫu Huấn Luyện')
    ax2.set_ylabel('Tốc Độ Học (ΔR²/Δsamples)')
    ax2.set_title('Phân Tích Tốc Độ Học')
    ax2.grid(True)
    ax2.legend(loc='best')
    
    # Thêm thông tin về tốc độ học
    ax2.text(0.05, 0.95,
             f'Tốc Độ Học TB (Training): {np.mean(train_slopes[1:]):.3e}\n'
             f'Tốc Độ Học TB (CV): {np.mean(test_slopes[1:]):.3e}\n'
             f'Tỷ Lệ Tốc Độ: {np.mean(train_slopes[1:])/np.mean(test_slopes[1:]):.2f}',
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_TRAINING_VIS_DIR, f'learning_curve_analysis.png'),
                dpi=DPI, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close()
    
    # Trả về thông tin phân tích
    return {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'gap_analysis': {
            'max_gap': np.max(gap),
            'min_gap': np.min(gap),
            'avg_gap': np.mean(gap)
        },
        'learning_rate': {
            'train_avg_rate': np.mean(train_slopes[1:]),
            'test_avg_rate': np.mean(test_slopes[1:]),
            'rate_ratio': np.mean(train_slopes[1:])/np.mean(test_slopes[1:])
        }
    }

def plot_combined_model_analysis(model, X_test, y_test, model_name):
    """
    Tạo biểu đồ ghép phân tích mô hình tốt nhất với cải tiến về hiển thị sai số
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # Tính toán các thống kê về sai số
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    residual_skew = stats.skew(residuals)
    residual_kurtosis = stats.kurtosis(residuals)
    
    # Tạo figure với 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Bỏ tiêu đề tổng thể để tránh chèn chữ
    
    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(y_test, y_pred, alpha=0.5, c='blue', s=30)  # Đổi sang màu xanh nước biển
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Giá Thực Tế')
    ax1.set_ylabel('Giá Dự Đoán')
    ax1.set_title('So Sánh Giá Thực Tế và Dự Đoán')
    
    # Thêm metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    ax1.text(0.05, 0.95, 
             f'MAE: {metrics["MAE"]:,.0f}$\n'
             f'RMSE: {metrics["RMSE"]:,.0f}$\n'
             f'R2: {metrics["R2"]:.3f}\n'
             f'MAPE: {metrics["MAPE"]:.1f}%',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals Distribution với đường chuẩn
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(residuals, kde=True, ax=ax2, color='blue')  # Đổi sang màu xanh nước biển
    
    # Thêm đường phân phối chuẩn
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x, residual_mean, residual_std)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, normal_pdf, 'r--', label='Phân Phối Chuẩn')
    ax2_twin.set_ylabel('Mật Độ Xác Suất')
    
    ax2.set_xlabel('Sai Số')
    ax2.set_ylabel('Số Lượng')
    ax2.set_title('Phân Phối Sai Số')
    
    # Thêm thống kê về phân phối
    ax2.text(0.05, 0.95,
             f'Trung Bình: {residual_mean:,.0f}$\n'
             f'Độ Lệch Chuẩn: {residual_std:,.0f}$\n'
             f'Độ Lệch: {residual_skew:.2f}\n'
             f'Độ Nhọn: {residual_kurtosis:.2f}',
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Residuals vs Predicted với cải tiến
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Vẽ scatter plot với màu xanh nước biển
    scatter = ax3.scatter(y_pred, residuals, alpha=0.5, c='blue', s=30)
    
    # Thêm đường hồi quy
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    ax3.plot(y_pred, p(y_pred), "r--", alpha=0.8, label=f'Xu Hướng (độ dốc={z[0]:.2e})')
    
    # Thêm đường ngang ở y=0 và các đường ±2*std
    ax3.axhline(y=0, color='r', linestyle='--', label='Đường Sai Số 0')
    ax3.axhline(y=2*residual_std, color='g', linestyle=':', label='±2*Độ Lệch Chuẩn')
    ax3.axhline(y=-2*residual_std, color='g', linestyle=':')
    
    # Thêm vùng tô màu cho ±2*std
    ax3.fill_between([y_pred.min(), y_pred.max()], 
                     -2*residual_std, 2*residual_std, 
                     alpha=0.1, color='g', label='Vùng ±2*Độ Lệch Chuẩn')
    
    ax3.set_xlabel('Giá Dự Đoán')
    ax3.set_ylabel('Sai Số')
    ax3.set_title('Sai Số Theo Giá Dự Đoán')
    ax3.legend()
    
    # 4. Feature Importance (nếu có)
    ax4 = fig.add_subplot(gs[1, 1])
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        feature_names = []
        for name, trans, cols in model.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                feature_names.extend(cols)  # CatBoostEncoder: chỉ cần tên cột gốc
        
        indices = np.argsort(importances)[::-1]
        top_n = 10  # Chỉ hiển thị top 10 features
        ax4.barh(range(top_n), importances[indices][:top_n], color='blue')  # Đổi sang màu xanh nước biển
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels([feature_names[i] for i in indices[:top_n]])
        ax4.set_xlabel('Độ Quan Trọng')
        ax4.set_title('Top 10 Đặc Trưng Quan Trọng Nhất')
    else:
        ax4.text(0.5, 0.5, 'Không có thông tin về độ quan trọng của đặc trưng\ncho loại mô hình này',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Độ Quan Trọng Đặc Trưng')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_TRAINING_VIS_DIR, f'combined_analysis_{model_name.lower().replace(" ", "_")}.png'),
                dpi=DPI, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close()
    
    # Trả về thông tin phân tích sai số
    return {
        'residual_stats': {
            'mean': residual_mean,
            'std': residual_std,
            'skewness': residual_skew,
            'kurtosis': residual_kurtosis,
            'trend_slope': z[0],
            'within_2std': np.mean(np.abs(residuals) <= 2*residual_std) * 100
        }
    }

def plot_model_comparison(results):
    """Vẽ biểu đồ so sánh các mô hình."""
    # Chuẩn bị dữ liệu
    model_names = list(results.keys())
    metrics = ['MAE', 'RMSE', 'R2']
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. So sánh metrics
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[name]['metrics'][metric] for name in model_names]
        if metric == 'R2':
            ax1.bar(x + i*width, values, width, label=metric)
            ax1.set_ylim(0, 1)
        else:
            ax1.bar(x + i*width, values, width, label=metric)
    
    ax1.set_title('So Sánh Hiệu Suất Các Mô Hình')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylabel('Giá Trị')
    
    # 2. So sánh thời gian huấn luyện
    training_times = [results[name]['training_time'] for name in model_names]
    ax2.bar(x, training_times, color='blue')  # Đổi sang màu xanh nước biển
    ax2.set_title('So Sánh Thời Gian Huấn Luyện')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Thời Gian (giây)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_TRAINING_VIS_DIR, 'model_comparison.png'),
                dpi=DPI, format=SAVE_FORMAT, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """
    Đánh giá chi tiết mô hình trên tập test.
    """
    y_pred = model.predict(X_test)
    
    # Tính các metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    # Tạo biểu đồ ghép và lấy thông tin phân tích sai số
    residual_analysis = plot_combined_model_analysis(model, X_test, y_test, model_name)
    
    # In thông tin phân tích sai số
    print("\nPhân tích sai số:")
    stats = residual_analysis['residual_stats']
    print(f"- Sai số trung bình: {stats['mean']:,.0f}$")
    print(f"- Độ lệch chuẩn: {stats['std']:,.0f}$")
    print(f"- Độ lệch (Skewness): {stats['skewness']:.2f}")
    print(f"- Độ nhọn (Kurtosis): {stats['kurtosis']:.2f}")
    print(f"- Độ dốc trend: {stats['trend_slope']:.2e}")
    print(f"- % điểm trong ±2*std: {stats['within_2std']:.1f}%")
    
    return metrics

def feature_importance_analysis(model, X_train, model_name):
    """
    Phân tích và hiển thị độ quan trọng của các features.
    """
    try:
        # Lấy tên features sau khi preprocess
        feature_names = []
        for name, trans, cols in model.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                feature_names.extend(cols)  # CatBoostEncoder: chỉ cần tên cột gốc
        
        # Lấy độ quan trọng của features
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
        elif hasattr(model.named_steps['model'], 'coef_'):
            importances = np.abs(model.named_steps['model'].coef_)
        else:
            logger.warning(f"Mô hình {model_name} không có thuộc tính feature_importances_ hoặc coef_")
            return
        
        # Sắp xếp features theo độ quan trọng
        indices = np.argsort(importances)[::-1]
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.tight_layout()
        
        # Lưu biểu đồ
        plt.savefig(os.path.join(MODEL_TRAINING_VIS_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'),
                    dpi=DPI, format=SAVE_FORMAT, bbox_inches='tight')
        plt.close()
        
        # In top features
        logger.info(f"\nTop 10 features quan trọng nhất của {model_name}:")
        for i in range(min(10, len(indices))):
            logger.info(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
    except Exception as e:
        logger.error(f"Lỗi khi phân tích độ quan trọng của features: {str(e)}")

def train_final_model(X_train, y_train, preprocessor, best_params):
    """
    Huấn luyện mô hình cuối cùng với các tham số tốt nhất.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    model = GradientBoostingRegressor(**best_params)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    try:
        # Import các module cần thiết
        from data_loader import load_data
        from preprocessing import run_preprocessing_steps
        
        print("\n=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ===\n")
        
        # 1. Load và tiền xử lý dữ liệu
        df = load_data()
        df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, _ = run_preprocessing_steps(df)
        print(f"Dữ liệu: {X_train.shape[0]:,} mẫu huấn luyện, {X_test.shape[0]:,} mẫu kiểm tra")
        
        # 2. So sánh các mô hình
        print("\n=== KẾT QUẢ CÁC MÔ HÌNH ===")
        preprocessor = preprocess_features(X_train, X_test, numeric_features, categorical_features)
        best_model_name, results = compare_base_models(X_train, y_train, X_test, y_test, preprocessor)
        
        # In kết quả ngắn gọn của từng mô hình
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n{name}:")
            print(f"R²: {metrics['R2']:.3f} | MAE: {metrics['MAE']:,.0f}$ | MAPE: {metrics['MAPE']:.1f}%")
        
        # 3. Kết quả mô hình tốt nhất
        print(f"\n=== MÔ HÌNH TỐT NHẤT: {best_model_name} ===")
        best_model = results[best_model_name]['pipeline']
        metrics = evaluate_model(best_model, X_test, y_test, best_model_name)
        
        print(f"\nMetrics:")
        print(f"R²: {metrics['R2']:.3f}")
        print(f"MAE: {metrics['MAE']:,.0f}$")
        print(f"MAPE: {metrics['MAPE']:.1f}%")
        
        # 4. Lưu mô hình
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"\nĐã lưu mô hình vào: {model_path}")
        print(f"Biểu đồ đánh giá được lưu trong: {MODEL_TRAINING_VIS_DIR}")
        
    except Exception as e:
        print(f"\nLỗi: {str(e)}")
        raise 