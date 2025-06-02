import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from config import *

logger = logging.getLogger(__name__)

def plot_boxplots(df, numeric_features):
    """Vẽ boxplot cho các features số."""
    logger.info("Bắt đầu vẽ boxplot cho các features số")
    
    for feature in numeric_features:
        plt.figure(figsize=FIGURE_SIZE)
        sns.boxplot(data=df, y=feature)
        plt.title(f'Boxplot của {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, f'boxplot_{feature}.png'), 
                    dpi=DPI, format=SAVE_FORMAT)
        plt.close()
    
    logger.info("Hoàn thành vẽ boxplot cho các features số")

def plot_scatter_matrix(df, numeric_features, target_column):
    """Vẽ ma trận scatter plot cho các features số."""
    logger.info("Bắt đầu vẽ ma trận scatter plot")
    
    # Chọn các features để vẽ
    features_to_plot = list(numeric_features) + [target_column]
    
    # Vẽ scatter matrix
    plt.figure(figsize=(FIGURE_SIZE[0]*2, FIGURE_SIZE[1]*2))
    sns.pairplot(df[features_to_plot], diag_kind='kde')
    plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, 'scatter_matrix.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    logger.info("Hoàn thành vẽ ma trận scatter plot")

def plot_target_distribution(df, target_column):
    """Vẽ phân phối của biến mục tiêu."""
    logger.info("Bắt đầu vẽ phân phối của biến mục tiêu")
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Vẽ histogram với KDE
    sns.histplot(data=df, x=target_column, kde=True)
    plt.title(f'Phân phối của {target_column}')
    plt.xlabel('Giá trị')
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, 'target_distribution.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    # Vẽ boxplot
    plt.figure(figsize=FIGURE_SIZE)
    sns.boxplot(data=df, y=target_column)
    plt.title(f'Boxplot của {target_column}')
    plt.ylabel('Giá trị')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, 'target_boxplot.png'), 
                dpi=DPI, format=SAVE_FORMAT)
    plt.close()
    
    logger.info("Hoàn thành vẽ phân phối của biến mục tiêu")

def plot_feature_relationships(df, numeric_features, target_column):
    """Vẽ biểu đồ thể hiện mối quan hệ giữa các features và biến mục tiêu."""
    logger.info("Bắt đầu vẽ biểu đồ mối quan hệ features-target")
    
    for feature in numeric_features:
        plt.figure(figsize=FIGURE_SIZE)
        
        # Vẽ scatter plot
        sns.scatterplot(data=df, x=feature, y=target_column, alpha=0.5)
        
        # Thêm đường hồi quy
        sns.regplot(data=df, x=feature, y=target_column, scatter=False, color='red')
        
        plt.title(f'Mối quan hệ giữa {feature} và {target_column}')
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, f'relationship_{feature}_{target_column}.png'), 
                    dpi=DPI, format=SAVE_FORMAT)
        plt.close()
    
    logger.info("Hoàn thành vẽ biểu đồ mối quan hệ features-target")

def plot_categorical_relationships(df, categorical_features, target_column):
    """Vẽ biểu đồ thể hiện mối quan hệ giữa các features phân loại và biến mục tiêu."""
    logger.info("Bắt đầu vẽ biểu đồ mối quan hệ categorical-target")
    
    for feature in categorical_features:
        plt.figure(figsize=FIGURE_SIZE)
        
        # Vẽ boxplot
        sns.boxplot(data=df, x=feature, y=target_column)
        plt.title(f'Mối quan hệ giữa {feature} và {target_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_VIS_DIR, f'categorical_relationship_{feature}_{target_column}.png'), 
                    dpi=DPI, format=SAVE_FORMAT)
        plt.close()
    
    logger.info("Hoàn thành vẽ biểu đồ mối quan hệ categorical-target")

def create_visualization_report(df, numeric_features, categorical_features, target_column):
    """Tạo báo cáo trực quan hóa đầy đủ."""
    logger.info("Bắt đầu tạo báo cáo trực quan hóa")
    
    # Vẽ các biểu đồ cơ bản
    plot_target_distribution(df, target_column)
    plot_boxplots(df, numeric_features)
    
    # Vẽ các biểu đồ phân tích mối quan hệ
    plot_scatter_matrix(df, numeric_features, target_column)
    plot_feature_relationships(df, numeric_features, target_column)
    plot_categorical_relationships(df, categorical_features, target_column)
    
    logger.info("Hoàn thành tạo báo cáo trực quan hóa") 