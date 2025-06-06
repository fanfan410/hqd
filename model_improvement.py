import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from hqd.preprocessing import run_preprocessing_steps, log_transform_data, create_training_preprocessor
from config import *

# Định nghĩa thư mục lưu kết quả cải tiến
IMPROVEMENT_VIS_DIR = os.path.join('visualizations', 'model_improvement')
os.makedirs(IMPROVEMENT_VIS_DIR, exist_ok=True)

# 1. Load dữ liệu và preprocessing
from hqd.preprocessing import run_preprocessing_steps, log_transform_data, create_training_preprocessor
from config import *

df = pd.read_csv('miami-housing.csv')
df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, _ = run_preprocessing_steps(df, verbose=False)
X_train, X_test, y_train, y_test = log_transform_data(X_train, X_test, y_train, y_test)

# Drop feature mạnh TRƯỚC khi tạo preprocessor
features_to_drop = ['PRICE_PER_SQFT', 'LIVING_LAND_RATIO', 'AVG_IMPORTANT_DIST']
X_train = X_train.drop(columns=[col for col in features_to_drop if col in X_train.columns])
X_test = X_test.drop(columns=[col for col in features_to_drop if col in X_test.columns])
print('Features sau khi drop:', list(X_train.columns))

# Tạo lại preprocessor từ X_train mới
preprocessor = create_training_preprocessor(
    X_train.select_dtypes(include=['int64', 'float64']).columns,
    X_train.select_dtypes(include=['object', 'category']).columns
)

# 2. Block cho các ý tưởng cải tiến
# --- Ví dụ: Feature Engineering mới ---
def improved_feature_engineering(df):
    # Thêm các biến mới, biến tương tác, biến phi tuyến, v.v.
    # df['new_feature'] = ...
    return df

# --- Ví dụ: Tuning Random Forest ---
def tune_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    param_grid = {
        'n_estimators': [50],
        'max_depth': [5],
        'min_samples_split': [20],
        'min_samples_leaf': [10],
        'max_features': ['log2']
    }
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    print('Best params:', grid.best_params_)
    print('Best RMSE:', -grid.best_score_)
    return grid.best_estimator_

# --- Ví dụ: Thử mô hình mới (XGBoost, LightGBM, CatBoost) ---
def try_xgboost(X_train, y_train):
    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def try_lightgbm(X_train, y_train):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def try_catboost(X_train, y_train):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

# --- Ví dụ: Stacking/Ensemble ---
def try_stacking(X_train, y_train):
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    estimators = [
        ('ridge', Ridge()),
        ('rf', RandomForestRegressor(random_state=RANDOM_STATE))
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stack.fit(X_train, y_train)
    return stack

# 3. Lưu và so sánh kết quả
# (Gợi ý: Tạo hàm evaluate_and_save để tính metrics, vẽ biểu đồ, lưu kết quả)
def evaluate_and_save(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    print(f"{model_name}: R2={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.0f} | MAE={metrics['MAE']:.0f} | MAPE={metrics['MAPE']:.2f}%")
    # Vẽ scatter actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, c='blue', s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá Thực Tế')
    plt.ylabel('Giá Dự Đoán')
    plt.title(f'So Sánh Giá Thực Tế và Dự Đoán - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(IMPROVEMENT_VIS_DIR, f'actual_vs_pred_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    return metrics


if __name__ == "__main__":
    # 1. Load dữ liệu
    df = pd.read_csv("miami-housing.csv")  # Đảm bảo DATA_PATH đúng đường dẫn file csv của bạn
    df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, preprocessor = run_preprocessing_steps(df, verbose=False)

    results = []
    successful_models = []

    # 2. Random Forest gốc (baseline)
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf_base = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=200,        # Từ tối ưu
                max_depth=6,             # Từ tối ưu
                min_samples_split=25,    # Từ tối ưu
                min_samples_leaf=12,     # Từ tối ưu
                max_features='sqrt',     # Từ tối ưu
                bootstrap=True,          # Từ tối ưu
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            ))
        ])
        rf_base.fit(X_train, y_train)
        metrics_rf_base = evaluate_and_save(rf_base, X_test, y_test, "RF_Baseline")
        results.append({"model": "RF_Baseline", **metrics_rf_base})
        successful_models.append("RF_Baseline")
    except Exception as e:
        print(f"Lỗi RF_Baseline: {e}")

    # 3. Tuning hyperparameters với Optuna
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 20, 40),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 20),
                'max_features': trial.suggest_categorical('max_features', ['log2'])
            }
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=N_JOBS))
            ])
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()
            return -score
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        print(f"Best params (Optuna): {best_params}")
        print(f"Best RMSE (Optuna): {-study.best_value:.2f}")
        rf_tuned = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ])
        rf_tuned.fit(X_train, y_train)
        metrics_rf_tuned = evaluate_and_save(rf_tuned, X_test, y_test, "RF_Tuned_Optuna")
        results.append({"model": "RF_Tuned_Optuna", **metrics_rf_tuned})
        successful_models.append("RF_Tuned_Optuna")
    except Exception as e:
        print(f"Lỗi RF_Tuned_Optuna: {e}")

    print("\n--- Đang chạy block SHAP (Feature Selection) ---")
    try:
        import shap
        rf_fs = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ])
        rf_fs.fit(X_train, y_train)
        X_shap = X_train.iloc[:500, :] if X_train.shape[0] > 500 else X_train
        explainer = shap.TreeExplainer(rf_fs.named_steps['model'])
        shap_values = explainer.shap_values(rf_fs.named_steps['preprocessor'].transform(X_shap))
        importances = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importances)[-15:]
        X_train_sel = X_train.iloc[:, top_idx]
        X_test_sel = X_test.iloc[:, top_idx]
        rf_fs_sel = Pipeline([
            ('preprocessor', create_training_preprocessor(
                X_train_sel.select_dtypes(include=['int64', 'float64']).columns,
                X_train_sel.select_dtypes(include=['object', 'category']).columns
            )),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ])
        rf_fs_sel.fit(X_train_sel, y_train)
        metrics_rf_fs = evaluate_and_save(rf_fs_sel, X_test_sel, y_test, "RF_SHAP_FeatureSelection")
        results.append({"model": "RF_SHAP_FeatureSelection", **metrics_rf_fs})
        successful_models.append("RF_SHAP_FeatureSelection")
    except Exception as e:
        print(f"Lỗi SHAP: {e}")

    print("\n--- Đang chạy block ExtraTrees ---")
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        et = Pipeline([
            ('preprocessor', preprocessor),
            ('model', ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1))
        ])
        et.fit(X_train, y_train)
        metrics_et = evaluate_and_save(et, X_test, y_test, "ExtraTrees")
        results.append({"model": "ExtraTrees", **metrics_et})
        successful_models.append("ExtraTrees")
    except Exception as e:
        print(f"Lỗi ExtraTrees: {e}")

    print("\n--- Đang chạy block Weighted Random Forest ---")
    try:
        sample_weight = np.where(y_train > y_train.median(), 2, 1)
        rf_weighted = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ])
        rf_weighted.fit(X_train, y_train, model__sample_weight=sample_weight)
        metrics_rf_weighted = evaluate_and_save(rf_weighted, X_test, y_test, "RF_Weighted")
        results.append({"model": "RF_Weighted", **metrics_rf_weighted})
        successful_models.append("RF_Weighted")
    except Exception as e:
        print(f"Lỗi RF_Weighted: {e}")

    print("\n--- Đang chạy block Deep Forest (gcForest) ---")
    try:
        from deepforest import CascadeForestRegressor
        gc = CascadeForestRegressor(random_state=RANDOM_STATE)
        gc.fit(X_train.values, y_train.values)
        metrics_gc = evaluate_and_save(gc, X_test.values, y_test.values, "DeepForest_gcForest")
        results.append({"model": "DeepForest_gcForest", **metrics_gc})
        successful_models.append("DeepForest_gcForest")
    except Exception as e:
        print(f"Lỗi Deep Forest: {e}")

    print("\n--- Đang chạy block Stacking ---")
    try:
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE)),
            ('et', ExtraTreesRegressor(random_state=RANDOM_STATE)),
            ('ridge', Ridge())
        ]
        stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
        stack.fit(X_train, y_train)
        metrics_stack = evaluate_and_save(stack, X_test, y_test, "Stacking_RF_ET_Ridge")
        results.append({"model": "Stacking_RF_ET_Ridge", **metrics_stack})
        successful_models.append("Stacking_RF_ET_Ridge")
    except Exception as e:
        print(f"Lỗi Stacking: {e}")

    print("\n--- Đang chạy block AutoML (AutoGluon) ---")
    try:
        from autogluon.tabular import TabularPredictor
        train_data = X_train.copy()
        train_data['target'] = y_train
        predictor = TabularPredictor(label='target', path=IMPROVEMENT_VIS_DIR).fit(train_data, time_limit=600)
        y_pred_automl = predictor.predict(X_test)
        metrics_automl = {
            'MAE': mean_absolute_error(y_test, y_pred_automl),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_automl)),
            'R2': r2_score(y_test, y_pred_automl),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred_automl) * 100
        }
        print(f"AutoML_AutoGluon: R2={metrics_automl['R2']:.4f} | RMSE={metrics_automl['RMSE']:.0f} | MAE={metrics_automl['MAE']:.0f} | MAPE={metrics_automl['MAPE']:.2f}%")
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred_automl, alpha=0.5, c='blue', s=30)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Giá Thực Tế')
        plt.ylabel('Giá Dự Đoán')
        plt.title('So Sánh Giá Thực Tế và Dự Đoán - AutoML')
        plt.tight_layout()
        plt.savefig(os.path.join(IMPROVEMENT_VIS_DIR, 'actual_vs_pred_AutoML.png'))
        plt.close()
        results.append({"model": "AutoML_AutoGluon", **metrics_automl})
        successful_models.append("AutoML_AutoGluon")
    except Exception as e:
        print(f"Lỗi AutoML: {e}")

    print("\n--- Đang chạy block RF với RobustScaler ---")
    try:
        rf_robust = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS))
        rf_robust.fit(X_train, y_train)
        metrics_rf_robust = evaluate_and_save(rf_robust, X_test, y_test, "RF_RobustScaler")
        results.append({"model": "RF_RobustScaler", **metrics_rf_robust})
        successful_models.append("RF_RobustScaler")
    except Exception as e:
        print(f"Lỗi RF_RobustScaler: {e}")

    print("\n--- Đang chạy block BaggingRegressor (ensemble RF) ---")
    try:
        from sklearn.ensemble import BaggingRegressor
        import sklearn
        bagging_kwargs = {
            'n_estimators': 10,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS
        }
        rf_base = RandomForestRegressor(n_estimators=10, max_depth=3, min_samples_split=10, min_samples_leaf=5, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS)
        # Tùy version sklearn, dùng estimator hoặc base_estimator
        try:
            bagging_rf = Pipeline([
                ('preprocessor', preprocessor),
                ('model', BaggingRegressor(estimator=rf_base, **bagging_kwargs))
            ])
        except TypeError:
            bagging_rf = Pipeline([
                ('preprocessor', preprocessor),
                ('model', BaggingRegressor(base_estimator=rf_base, **bagging_kwargs))
            ])
        bagging_rf.fit(X_train, y_train)
        metrics_bagging = evaluate_and_save(bagging_rf, X_test, y_test, "Bagging_RF")
        results.append({"model": "Bagging_RF", **metrics_bagging})
        successful_models.append("Bagging_RF")
    except Exception as e:
        print(f"Lỗi Bagging_RF: {e}")

    print("\n--- Đang chạy block VotingRegressor (ensemble RF/ET) ---")
    try:
        from sklearn.ensemble import VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
        voting = Pipeline([
            ('preprocessor', preprocessor),
            ('model', VotingRegressor(estimators=[
                ('rf1', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS)),
                ('rf2', RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_STATE+1, n_jobs=N_JOBS)),
                ('et1', ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE+2, n_jobs=N_JOBS))
            ]))
        ])
        voting.fit(X_train, y_train)
        metrics_voting = evaluate_and_save(voting, X_test, y_test, "Voting_RF_ET")
        results.append({"model": "Voting_RF_ET", **metrics_voting})
        successful_models.append("Voting_RF_ET")
    except Exception as e:
        print(f"Lỗi Voting_RF_ET: {e}")

    print("\n--- Đang chạy block Stacking nhiều RF với meta-learner là Ridge ---")
    try:
        from sklearn.ensemble import StackingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        estimators = [
            ('rf1', RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=20, min_samples_leaf=10, max_features='log2', random_state=RANDOM_STATE, n_jobs=N_JOBS)),
            ('rf2', RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_STATE+1, n_jobs=N_JOBS)),
            ('rf3', RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE+2, n_jobs=N_JOBS))
        ]
        stacking_rf = Pipeline([
            ('preprocessor', preprocessor),
            ('model', StackingRegressor(estimators=estimators, final_estimator=Ridge()))
        ])
        stacking_rf.fit(X_train, y_train)
        metrics_stacking_rf = evaluate_and_save(stacking_rf, X_test, y_test, "Stacking_MultiRF_Ridge")
        results.append({"model": "Stacking_MultiRF_Ridge", **metrics_stacking_rf})
        successful_models.append("Stacking_MultiRF_Ridge")
    except Exception as e:
        print(f"Lỗi Stacking_MultiRF_Ridge: {e}")

    # Lưu bảng tổng hợp kết quả
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(IMPROVEMENT_VIS_DIR, 'improvement_results.csv'), index=False)
    print("\n==== DANH SÁCH MÔ HÌNH CẢI TIẾN CHẠY THÀNH CÔNG ====")
    print(", ".join(successful_models))
    print("\n==== BẢNG SO SÁNH CÁC MÔ HÌNH CẢI TIẾN ====")
    print("| {:<25} | {:>7} | {:>10} | {:>10} | {:>7} |".format('Model', 'R2', 'RMSE', 'MAE', 'MAPE'))
    print("|" + "-"*25 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*8 + "|")
    for _, row in df_results.sort_values('R2', ascending=False).iterrows():
        print("| {:<25} | {:>7.4f} | {:>10,.0f} | {:>10,.0f} | {:>6.2f}% |".format(
            row['model'], row['R2'], row['RMSE'], row['MAE'], row['MAPE']))
    print("\nBảng tổng hợp kết quả các mô hình cải tiến đã lưu vào improvement_results.csv") 