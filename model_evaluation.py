# Sau khi luyện mô hình với tham số tốt nhất và trước khi đánh giá trên tập test

# 4.5. VẼ LEARNING CURVE
print("\n4.5. VẼ LEARNING CURVE")
from sklearn.model_selection import learning_curve

# Vẽ Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_processed, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Tính RMSE
train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse = np.sqrt(-val_scores.mean(axis=1))
train_rmse_std = np.sqrt(-train_scores.std(axis=1))
val_rmse_std = np.sqrt(-val_scores.std(axis=1))

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_rmse, label='Training RMSE')
plt.plot(train_sizes, val_rmse, label='Validation RMSE')
plt.fill_between(train_sizes, train_rmse - train_rmse_std, train_rmse + train_rmse_std, alpha=0.1)
plt.fill_between(train_sizes, val_rmse - val_rmse_std, val_rmse + val_rmse_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve của Gradient Boosting (Sau hiệu chỉnh)')
plt.legend()
plt.savefig('learning_curve.png')
plt.close()

# 5. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
# ... tiếp tục với phần còn lại của code ... 