import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from config import *
from typing import Dict, Tuple, List, Union
import warnings
import sys
import io

# Thiết lập encoding cho stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập logging với UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log', encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Tắt các warning không cần thiết
warnings.filterwarnings('ignore')

# Định nghĩa các ngưỡng đánh giá
METRICS_THRESHOLDS = {
    'MAE': {
        'good': 50000,
        'medium': 100000,
        'poor': float('inf')
    },
    'RMSE': {
        'good': 60000,
        'medium': 120000,
        'poor': float('inf')
    },
    'R2': {
        'excellent': 0.9,
        'good': 0.8,
        'medium': 0.7,
        'poor': float('-inf')
    },
    'MAPE': {
        'good': 15,
        'medium': 25,
        'poor': float('inf')
    }
}

PRICE_RANGES = {
    'low': (0, 200000),
    'medium': (200000, 500000),
    'high': (500000, 1000000),
    'very_high': (1000000, float('inf'))
}

def calculate_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các metrics về độ chính xác của mô hình.
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        Dict chứa các metrics: MAE, RMSE, R2, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Tính MAPE, tránh chia cho 0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def evaluate_metrics_performance(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Đánh giá hiệu suất dựa trên các metrics và ngưỡng đã định nghĩa.
    
    Args:
        metrics: Dict chứa các metrics đã tính
        
    Returns:
        Dict chứa đánh giá cho từng metric
    """
    evaluations = {}
    
    for metric, value in metrics.items():
        if metric in METRICS_THRESHOLDS:
            thresholds = METRICS_THRESHOLDS[metric]
            if metric == 'R2':  # R2 có thang đánh giá ngược
                if value >= thresholds['excellent']:
                    evaluations[metric] = 'Xuất sắc'
                elif value >= thresholds['good']:
                    evaluations[metric] = 'Tốt'
                elif value >= thresholds['medium']:
                    evaluations[metric] = 'Trung bình'
                else:
                    evaluations[metric] = 'Cần cải thiện'
            else:
                if value <= thresholds['good']:
                    evaluations[metric] = 'Tốt'
                elif value <= thresholds['medium']:
                    evaluations[metric] = 'Trung bình'
                else:
                    evaluations[metric] = 'Cần cải thiện'
    
    return evaluations

def measure_prediction_time(model, X_test: np.ndarray) -> Dict[str, float]:
    """
    Đo thời gian dự đoán của mô hình.
    
    Args:
        model: Mô hình cần đánh giá
        X_test: Dữ liệu test
        
    Returns:
        Dict chứa thời gian dự đoán tổng và trung bình trên mỗi mẫu
    """
    n_samples = len(X_test)
    
    # Đo thời gian dự đoán
    start_time = time.time()
    _ = model.predict(X_test)
    total_time = time.time() - start_time
    
    # Tính thời gian trung bình trên mỗi mẫu (ms)
    avg_time_per_sample = (total_time / n_samples) * 1000
    
    return {
        'total_prediction_time': total_time,
        'avg_time_per_sample_ms': avg_time_per_sample
    }

def evaluate_prediction_speed(prediction_time: float) -> str:
    """
    Đánh giá tốc độ dự đoán của mô hình.
    
    Args:
        prediction_time: Thời gian dự đoán trung bình trên mỗi mẫu (ms)
        
    Returns:
        Đánh giá tốc độ
    """
    if prediction_time < 100:
        return 'Rất nhanh (phù hợp cho ứng dụng thời gian thực)'
    elif prediction_time < 1000:
        return 'Nhanh (phù hợp cho ứng dụng web)'
    elif prediction_time < 5000:
        return 'Trung bình (phù hợp cho xử lý theo lô)'
    else:
        return 'Chậm (cần tối ưu hóa)'

def calculate_prediction_stability(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán độ ổn định của dự đoán dựa trên phân tích residuals.
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        Dict chứa các metrics về độ ổn định
    """
    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Tính độ lệch chuẩn của residuals
    std_residuals = np.std(residuals)
    
    # Tính tỷ lệ std/residuals để đánh giá độ ổn định
    stability_ratio = std_residuals / rmse if rmse != 0 else float('inf')
    
    return {
        'std_residuals': std_residuals,
        'stability_ratio': stability_ratio
    }

def evaluate_price_range_reliability(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá độ tin cậy của mô hình trên các khoảng giá khác nhau.
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        Dict chứa các metrics cho từng khoảng giá
    """
    reliability_metrics = {}
    
    for range_name, (min_price, max_price) in PRICE_RANGES.items():
        # Lọc dữ liệu trong khoảng giá
        mask = (y_true >= min_price) & (y_true < max_price)
        if np.sum(mask) > 0:  # Chỉ tính nếu có dữ liệu trong khoảng
            y_true_range = y_true[mask]
            y_pred_range = y_pred[mask]
            
            # Tính các metrics cho khoảng giá này
            mae = mean_absolute_error(y_true_range, y_pred_range)
            mape = np.mean(np.abs((y_true_range - y_pred_range) / y_true_range)) * 100
            
            reliability_metrics[range_name] = {
                'MAE': mae,
                'MAPE': mape,
                'sample_count': np.sum(mask)
            }
    
    return reliability_metrics

def calculate_composite_score(metrics: Dict[str, float], 
                            prediction_time: float,
                            stability_ratio: float) -> float:
    """
    Tính điểm đánh giá tổng hợp dựa trên các tiêu chí.
    
    Args:
        metrics: Dict chứa các metrics đã tính
        prediction_time: Thời gian dự đoán trung bình (ms)
        stability_ratio: Tỷ lệ độ ổn định
        
    Returns:
        Điểm tổng hợp (0-100)
    """
    # Tính điểm độ chính xác (60%)
    accuracy_score = 0
    if 'R2' in metrics:
        r2_score = min(max(metrics['R2'], 0), 1) * 100  # Chuyển R2 về thang 0-100
        accuracy_score = r2_score * 0.6
    
    # Tính điểm tốc độ (20%)
    speed_score = 0
    if prediction_time < 100:
        speed_score = 100
    elif prediction_time < 1000:
        speed_score = 80
    elif prediction_time < 5000:
        speed_score = 60
    else:
        speed_score = 40
    speed_score *= 0.2
    
    # Tính điểm độ ổn định (20%)
    stability_score = 0
    if stability_ratio < 0.5:
        stability_score = 100
    elif stability_ratio < 0.8:
        stability_score = 80
    elif stability_ratio < 1.2:
        stability_score = 60
    else:
        stability_score = 40
    stability_score *= 0.2
    
    # Tổng điểm
    total_score = accuracy_score + speed_score + stability_score
    
    return total_score

def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình trên tập test."""
    logger.info("Bắt đầu đánh giá mô hình trên tập test")
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính các metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log kết quả
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"R2 Score: {r2:.4f}")
    
    # Vẽ biểu đồ so sánh giá trị thực tế và dự đoán
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('So sánh giá trị thực tế và dự đoán')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'prediction_vs_actual.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # Vẽ biểu đồ phân phối sai số
    residuals = y_test - y_pred
    plt.figure(figsize=FIGURE_SIZE)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Sai số')
    plt.ylabel('Tần suất')
    plt.title('Phân phối sai số')
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'residuals_distribution.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'residuals': residuals
    }

def analyze_feature_importance(model, X_test, y_test, feature_names):
    """Phân tích tầm quan trọng của các features."""
    logger.info("Bắt đầu phân tích tầm quan trọng của features")
    
    # Tính permutation importance
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    
    # Tạo DataFrame kết quả
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Vẽ biểu đồ
    plt.figure(figsize=FIGURE_SIZE)
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Tầm quan trọng của các features')
    plt.xlabel('Tầm quan trọng')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'feature_importance.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    logger.info("Hoàn thành phân tích tầm quan trọng của features")
    return importance_df

def analyze_prediction_errors(model, X_test, y_test, feature_names):
    """Phân tích các trường hợp dự đoán sai."""
    logger.info("Bắt đầu phân tích các trường hợp dự đoán sai")
    
    # Dự đoán và tính sai số
    y_pred = model.predict(X_test)
    errors = np.abs(y_test - y_pred)
    
    # Tìm các trường hợp có sai số lớn nhất
    worst_cases_idx = np.argsort(errors)[-10:]  # Top 10 trường hợp sai nhiều nhất
    
    # Tạo DataFrame cho các trường hợp sai nhiều nhất
    worst_cases = pd.DataFrame({
        'actual': y_test.iloc[worst_cases_idx].values,
        'predicted': y_pred[worst_cases_idx],
        'error': errors.iloc[worst_cases_idx].values
    })
    
    # Thêm các features cho các trường hợp này
    for feature in feature_names:
        worst_cases[feature] = X_test.iloc[worst_cases_idx][feature].values
    
    # Log thông tin
    logger.info("\nTop 10 trường hợp dự đoán sai nhiều nhất:")
    logger.info(worst_cases[['actual', 'predicted', 'error']].to_string())
    
    # Vẽ biểu đồ phân tích các trường hợp sai
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(y_test, y_pred, alpha=0.5, label='Tất cả các trường hợp')
    plt.scatter(y_test.iloc[worst_cases_idx], y_pred[worst_cases_idx], 
               color='red', label='Top 10 trường hợp sai nhiều nhất')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('Phân tích các trường hợp dự đoán sai')
    plt.legend()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'worst_predictions.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    return worst_cases

def generate_evaluation_report(model, X_test, y_test, feature_names):
    """
    Tạo báo cáo đánh giá đầy đủ cho mô hình.
    
    Args:
        model: Mô hình cần đánh giá
        X_test: Dữ liệu test
        y_test: Giá trị thực tế
        feature_names: Tên các features
        
    Returns:
        Dict chứa báo cáo đánh giá với các metrics:
        - accuracy_metrics: Các metrics về độ chính xác
        - speed_metrics: Các metrics về tốc độ
        - stability_metrics: Các metrics về độ ổn định
        - price_range_reliability: Độ tin cậy theo khoảng giá
        - composite_score: Điểm đánh giá tổng hợp
        - feature_importance: Tầm quan trọng của features
        - worst_cases: Các trường hợp dự đoán sai nhiều nhất
    """
    # Tính toán các metrics
    y_pred = model.predict(X_test)
    accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred)
    metrics_evaluation = evaluate_metrics_performance(accuracy_metrics)
    prediction_time_metrics = measure_prediction_time(model, X_test)
    speed_evaluation = evaluate_prediction_speed(prediction_time_metrics['avg_time_per_sample_ms'])
    stability_metrics = calculate_prediction_stability(y_test, y_pred)
    price_reliability = evaluate_price_range_reliability(y_test, y_pred)
    composite_score = calculate_composite_score(
        accuracy_metrics,
        prediction_time_metrics['avg_time_per_sample_ms'],
        stability_metrics['stability_ratio']
    )
    importance_df = analyze_feature_importance(model, X_test, y_test, feature_names)
    worst_cases = analyze_prediction_errors(model, X_test, y_test, feature_names)
    
    # Tạo báo cáo tổng hợp
    report = {
        'accuracy_metrics': {
            'values': accuracy_metrics,
            'evaluation': metrics_evaluation
        },
        'speed_metrics': {
            'values': prediction_time_metrics,
            'evaluation': speed_evaluation
        },
        'stability_metrics': stability_metrics,
        'price_range_reliability': price_reliability,
        'composite_score': composite_score,
        'feature_importance': importance_df,
        'worst_cases': worst_cases
    }
    
    # Vẽ các biểu đồ đánh giá
    visualize_evaluation_results(report, y_test, y_pred)
    
    return report

def visualize_evaluation_results(report: Dict, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Vẽ các biểu đồ đánh giá kết quả.
    
    Args:
        report: Báo cáo đánh giá
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
    """
    # 1. Biểu đồ so sánh giá trị thực tế và dự đoán
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('So sánh giá trị thực tế và dự đoán')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'prediction_vs_actual.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # 2. Biểu đồ phân phối sai số
    residuals = y_true - y_pred
    plt.figure(figsize=FIGURE_SIZE)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Sai số')
    plt.ylabel('Tần suất')
    plt.title('Phân phối sai số')
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'residuals_distribution.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # 3. Biểu đồ đánh giá theo khoảng giá
    price_ranges = report['price_range_reliability']
    ranges = list(price_ranges.keys())
    mape_values = [metrics['MAPE'] for metrics in price_ranges.values()]
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(ranges, mape_values)
    plt.xlabel('Khoảng giá')
    plt.ylabel('MAPE (%)')
    plt.title('Đánh giá MAPE theo khoảng giá')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'price_range_mape.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # 4. Biểu đồ radar cho điểm đánh giá tổng hợp
    metrics = ['Độ chính xác', 'Tốc độ', 'Độ ổn định']
    scores = [
        report['accuracy_metrics']['values']['R2'] * 100,
        (100 - min(report['speed_metrics']['values']['avg_time_per_sample_ms'] / 50, 100)),
        (100 - min(report['stability_metrics']['stability_ratio'] * 50, 100))
    ]
    
    # Tính toán góc cho biểu đồ radar
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    scores = np.concatenate((scores, [scores[0]]))  # Đóng biểu đồ
    angles = np.concatenate((angles, [angles[0]]))  # Đóng biểu đồ
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scores)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    plt.title('Biểu đồ đánh giá tổng hợp')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_EVALUATION_VIS_DIR, 'composite_score_radar.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()

if __name__ == "__main__":
    try:
        print("\n=== TIÊU CHÍ ĐÁNH GIÁ MÔ HÌNH DỰ ĐOÁN GIÁ NHÀ ===")
        
        print("\n1. TIÊU CHÍ VỀ ĐỘ CHÍNH XÁC")
        print("----------------------------")
        print("1.1. Mean Absolute Error (MAE)")
        print("    - Định nghĩa: Sai số tuyệt đối trung bình giữa giá trị dự đoán và thực tế")
        print("    - Ngưỡng đánh giá:")
        print("      + Tốt: <= 50.000$")
        print("      + Trung bình: <= 100.000$")
        print("      + Cần cải thiện: > 100.000$")
        
        print("\n1.2. Root Mean Squared Error (RMSE)")
        print("    - Định nghĩa: Căn bậc hai của sai số bình phương trung bình")
        print("    - Ngưỡng đánh giá:")
        print("      + Tốt: <= 60.000$")
        print("      + Trung bình: <= 120.000$")
        print("      + Cần cải thiện: > 120.000$")
        
        print("\n1.3. R-squared (R²)")
        print("    - Định nghĩa: Tỷ lệ phương sai được giải thích bởi mô hình")
        print("    - Ngưỡng đánh giá:")
        print("      + Xuất sắc: >= 0,9")
        print("      + Tốt: >= 0,8")
        print("      + Trung bình: >= 0,7")
        print("      + Cần cải thiện: < 0,7")
        
        print("\n1.4. Mean Absolute Percentage Error (MAPE)")
        print("    - Định nghĩa: Sai số phần trăm tuyệt đối trung bình")
        print("    - Ngưỡng đánh giá:")
        print("      + Tốt: <= 15%")
        print("      + Trung bình: <= 25%")
        print("      + Cần cải thiện: > 25%")
        
        print("\n2. TIÊU CHÍ VỀ TỐC ĐỘ")
        print("----------------------")
        print("2.1. Thời gian dự đoán trung bình trên mỗi mẫu")
        print("    - Ngưỡng đánh giá:")
        print("      + Rất nhanh (< 100ms): Phù hợp cho ứng dụng thời gian thực")
        print("      + Nhanh (< 1000ms): Phù hợp cho ứng dụng web")
        print("      + Trung bình (< 5000ms): Phù hợp cho xử lý theo lô")
        print("      + Chậm (>= 5000ms): Cần tối ưu hóa")
        
        print("\n3. TIÊU CHÍ VỀ KHẢ NĂNG ỨNG DỤNG VÀ ĐỘ TIN CẬY")
        print("--------------------------------------------")
        print("3.1. Độ ổn định của dự đoán")
        print("    - Định nghĩa: Tỷ lệ độ lệch chuẩn của residuals trên RMSE")
        print("    - Ngưỡng đánh giá:")
        print("      + Tốt: < 0,5")
        print("      + Khá: < 0,8")
        print("      + Trung bình: < 1,2")
        print("      + Cần cải thiện: >= 1,2")
        
        print("\n3.2. Độ tin cậy theo khoảng giá")
        print("    - Khoảng giá đánh giá:")
        print("      + Thấp: 0 - 200.000$")
        print("      + Trung bình: 200.000$ - 500.000$")
        print("      + Cao: 500.000$ - 1.000.000$")
        print("      + Rất cao: > 1.000.000$")
        print("    - Đánh giá dựa trên MAPE cho từng khoảng giá")
        
        print("\n4. ĐIỂM ĐÁNH GIÁ TỔNG HỢP")
        print("-------------------------")
        print("4.1. Thang điểm: 0-100")
        print("    - Độ chính xác (R²): 60%")
        print("    - Tốc độ dự đoán: 20%")
        print("    - Độ ổn định: 20%")
        
        print("\nCác hàm đánh giá có sẵn trong module:")
        print("1. calculate_accuracy_metrics    - Tính toán các metrics về độ chính xác")
        print("2. evaluate_metrics_performance  - Đánh giá hiệu suất dựa trên các metrics")
        print("3. measure_prediction_time      - Đo thời gian dự đoán")
        print("4. evaluate_prediction_speed    - Đánh giá tốc độ dự đoán")
        print("5. calculate_prediction_stability - Tính toán độ ổn định")
        print("6. evaluate_price_range_reliability - Đánh giá độ tin cậy theo khoảng giá")
        print("7. calculate_composite_score    - Tính điểm đánh giá tổng hợp")
        print("8. analyze_feature_importance   - Phân tích tầm quan trọng của features")
        print("9. analyze_prediction_errors    - Phân tích các trường hợp dự đoán sai")
        print("10. generate_evaluation_report  - Tạo báo cáo đánh giá đầy đủ")
        print("11. visualize_evaluation_results - Tạo các biểu đồ đánh giá")
        
    except Exception as e:
        print(f"\nLỗi: {str(e)}")
        raise 