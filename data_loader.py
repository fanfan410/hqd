import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from config import *

# Thiết lập logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    filename=LOG_FILE
)
logger = logging.getLogger(__name__)

def load_data(filepath=DATA_FILE):
    """Đọc dữ liệu từ file CSV."""
    logger.info(f"Đang đọc dữ liệu từ {filepath}")
    try:
        df = pd.read_csv(filepath)
        
        # Kiểm tra cột target
        if ORIGINAL_TARGET not in df.columns:
            raise ValueError(f"Cột {ORIGINAL_TARGET} không tồn tại trong dữ liệu. "
                           f"Các cột hiện có: {df.columns.tolist()}")
        
        logger.info(f"Đã đọc dữ liệu thành công. Kích thước: {df.shape}")
        logger.info(f"Các cột trong dữ liệu: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
        raise

def explore_data(df):
    """Thực hiện khám phá dữ liệu cơ bản (EDA)."""
    logger.info("Bắt đầu khám phá dữ liệu (EDA)")
    
    try:
        # In thông tin cơ bản
        print("\n=== THÔNG TIN DỮ LIỆU ===")
        print(f"Kích thước: {df.shape}")
        print("\nKiểu dữ liệu:")
        print(df.dtypes)
        
        # Thống kê mô tả
        print("\n=== THỐNG KÊ MÔ TẢ ===")
        print(df.describe())
        
        # Vẽ phân phối giá nhà gốc
        if ORIGINAL_TARGET in df.columns:
            plt.figure(figsize=FIGURE_SIZE)
            sns.histplot(df[ORIGINAL_TARGET], kde=True)
            plt.title('Phân phối giá nhà')
            plt.savefig(os.path.join(DATA_LOADER_VIS_DIR, 'price_distribution_original.png'), 
                        dpi=DPI, format=SAVE_FORMAT)
            plt.close()
        
        # Vẽ ma trận tương quan
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            correlation = df[numeric_cols].corr()
            plt.figure(figsize=(14, 12))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
            plt.title('Ma trận tương quan')
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_LOADER_VIS_DIR, 'correlation_matrix.png'), 
                        dpi=DPI, format=SAVE_FORMAT)
            plt.close()
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình EDA: {str(e)}")
        print(f"\nLỗi: {str(e)}")
        raise
    
    logger.info("Hoàn thành EDA")

def analyze_missing_values(df):
    """In ra bảng gồm tên cột và số lượng giá trị thiếu cho từng cột."""
    logger.info("Bắt đầu phân tích giá trị thiếu")
    
    # Tính số lượng giá trị thiếu cho mỗi cột
    null_counts = df.isnull().sum()
    
    # Tạo DataFrame kết quả
    missing_data = pd.DataFrame({
        'Số lượng giá trị thiếu': null_counts
    })
    
    print("\nBẢNG GIÁ TRỊ THIẾU THEO CỘT:")
    print(missing_data)
    
    logger.info("Hoàn thành phân tích giá trị thiếu")
    return missing_data

def analyze_numeric_features(df):
    """Phân tích chi tiết các biến số."""
    logger.info("Bắt đầu phân tích các biến số")
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Vẽ boxplot cho các biến số
    plt.figure(figsize=(15, 10))
    df[numeric_cols].boxplot()
    plt.title('Boxplot của các biến số')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_LOADER_VIS_DIR, 'numeric_features_boxplot.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # Vẽ phân phối cho từng biến số
    for col in numeric_cols:
        plt.figure(figsize=FIGURE_SIZE)
        sns.histplot(df[col], kde=True)
        plt.title(f'Phân phối của {col}')
        plt.savefig(os.path.join(DATA_LOADER_VIS_DIR, f'numeric_{col}_distribution.png'), 
                    dpi=DPI, format=SAVE_FORMAT)
        plt.close()
    
    logger.info("Hoàn thành phân tích các biến số")

def analyze_categorical_features(df):
    """Phân tích chi tiết các biến phân loại."""
    logger.info("Bắt đầu phân tích các biến phân loại")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            plt.figure(figsize=FIGURE_SIZE)
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar')
            plt.title(f'Phân phối của {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_LOADER_VIS_DIR, f'categorical_{col}_distribution.png'), 
                        dpi=DPI, format=SAVE_FORMAT)
            plt.close()
            
            print(f"\nThống kê cho {col}:")
            print(value_counts)
    else:
        print("\nKhông có biến phân loại trong dữ liệu.")
    
    logger.info("Hoàn thành phân tích các biến phân loại")

def analyze_target_distribution(df, target_column='SALE_PRC', vis_dir=DATA_LOADER_VIS_DIR):
    """Phân tích và vẽ biểu đồ phân phối biến mục tiêu (giá bán nhà)."""
    print(f"\n=== PHÂN TÍCH BIẾN MỤC TIÊU: {target_column} ===")
    if target_column not in df.columns:
        print(f"Không tìm thấy cột {target_column} trong dữ liệu!")
        return
    # Thống kê mô tả
    print(df[target_column].describe())
    # Vẽ biểu đồ phân phối
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], bins=80, kde=True, color='steelblue', edgecolor=None)
    plt.title(f'Phân phối giá bán nhà ({target_column})')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'{target_column.lower()}_distribution.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Đã lưu biểu đồ phân phối tại: {save_path}")

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "miami-housing.csv"
    df = load_data(filepath)
    # Nếu muốn test tích hợp, chỉ import và gọi run_preprocessing_steps ở đây:
    # from preprocessing import run_preprocessing_steps
    # df = run_preprocessing_steps(df)
    explore_data(df)
    analyze_missing_values(df)
    analyze_target_distribution(df, target_column='SALE_PRC', vis_dir=DATA_LOADER_VIS_DIR) 