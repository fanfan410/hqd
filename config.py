import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os

# Cấu hình dữ liệu
DATA_FILE = 'miami-housing.csv'
TARGET_COLUMN = 'SALE_PRC_CLEANED'
ID_COLUMN = 'PARCELNO'
ORIGINAL_TARGET = 'SALE_PRC'

# Cấu hình đường dẫn hình ảnh
VISUALIZATIONS_DIR = 'visualizations'
DATA_LOADER_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, 'data_loader')
PREPROCESSING_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, 'preprocessing')
MODEL_TRAINING_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, 'model_training')
MODEL_EVALUATION_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, 'model_evaluation')
VISUALIZATION_VIS_DIR = os.path.join(VISUALIZATIONS_DIR, 'visualization')

# Tạo thư mục nếu chưa tồn tại
for dir_path in [VISUALIZATIONS_DIR, DATA_LOADER_VIS_DIR, PREPROCESSING_VIS_DIR, 
                MODEL_TRAINING_VIS_DIR, MODEL_EVALUATION_VIS_DIR, VISUALIZATION_VIS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Cấu hình model
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
GRID_SEARCH_CV_FOLDS = 3
N_JOBS = -1  # Sử dụng tất cả CPU cores

# Cấu hình hyperparameter tuning cho Gradient Boosting
GRID_SEARCH_PARAMS = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5],
    'model__min_samples_split': [5, 10],
    'model__subsample': [0.8, 1.0]
}

# Định nghĩa base models
BASE_MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE)
}

# Cấu hình visualization
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FORMAT = 'png'

# Cấu hình logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'housing_prediction.log' 