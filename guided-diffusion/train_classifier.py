"""
train_classifier.py 
Strategy: 
  Concatenates 5 Trajectories:
  1. Mean Consistency (Global Stability)
  2. Min Consistency (Structural Breakdown)
  3. Std Consistency (Noise Uniformity)
  4. Avg Abs Skewness (Gaussianity / Symmetry)
  5. Avg Abs Kurtosis (Gaussianity / Tails)
  + Physics Features on Mean Trajectory
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
# 请确保这里指向您存放 .npy 文件的实际路径
REAL_DIR = "../results/LSUN/real"  
FAKE_DIR = "../results/LSUN/fake"
# ===========================================

def extract_physics_features(trajectory):
    """
    仅针对 Mean 轨迹提取物理特征 
    这些特征描述了全局一致性随时间下降的速率和形态
    """
    if len(trajectory) < 5: return [0, 0, 0]
    
    # 1. 早期斜率 (Early Slope): 反应初始崩塌速度
    slope_early = (trajectory[min(5, len(trajectory)-1)] - trajectory[0]) / 5.0
    
    # 2. 曲线下面积 (Area Under Curve): 反应整体鲁棒性
    area = np.trapz(trajectory)
    
    # 3. 半衰期 (Half-life Index): 什么时候降到 0.5?
    # 注意：对于归一化后的 Mean 轨迹，通常从 1.0 开始
    half_life_idx = np.where(trajectory < 0.5)[0]
    half_life = half_life_idx[0] if len(half_life_idx) > 0 else len(trajectory)
    
    return [slope_early, area, half_life]

def load_data_ensemble(data_dir, label_val):
    """
    加载并融合 5 维特征轨迹
    """
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not npy_files:
        print(f"[Warn] No files found in {data_dir}")
        return None, None

    data_list = []
    print(f"Loading {len(npy_files)} files from {data_dir}...")
    
    skipped = 0
    for f in npy_files:
        try:
            # raw shape: [Time, 5] (New) or [Time, 3] (Old)
            raw = np.load(f)
            
            # 兼容性检查
            if raw.ndim != 2:
                skipped += 1
                continue
            
            rows, cols = raw.shape
            
            # --- 1. 分离通道 ---
            mean_traj = raw[:, 0]
            min_traj  = raw[:, 1]
            std_traj  = raw[:, 2]
            
            if cols >= 5:
                skew_traj = raw[:, 3]
                kurt_traj = raw[:, 4]
            else:
                skew_traj = np.zeros_like(mean_traj)
                kurt_traj = np.zeros_like(mean_traj)

            # --- 2. 归一化 (Normalization) ---
            mean_norm = mean_traj / (mean_traj[0] + 1e-6)
            min_norm  = min_traj  / (min_traj[0]  + 1e-6)
            std_norm  = std_traj  # Std 保持原值或归一化均可，这里保持原值以保留量级差异
            
            # --- 3. 特征融合 (Concatenation) ---
            # [Mean_0..T, Min_0..T, Std_0..T, Skew_0..T, Kurt_0..T]
            combined_vector = np.concatenate([
                mean_norm, 
                min_norm, 
                std_norm, 
                skew_traj, 
                kurt_traj
            ])
            
            # --- 4. 加入物理特征 ---
            phys_feats = extract_physics_features(mean_norm)
            
            # 最终特征向量
            final_features = np.concatenate([combined_vector, phys_feats])
            
            data_list.append(final_features)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            skipped += 1
            pass
            
    if skipped > 0:
        print(f"[Info] Skipped {skipped} invalid/corrupted files.")

    if not data_list: return None, None
    
    # 堆叠成矩阵 [N_samples, Feature_Dim]
    return np.stack(data_list, axis=0), np.full(len(data_list), label_val)

def main():
    print("=== Deepfake Trajectory Classifier V5 (Hybrid Ensemble) ===")
    print("Strategy: Consistency (Mean/Min/Std) + Gaussianity (Skew/Kurt)")
    
    # 1. 加载数据
    print(f"Loading REAL data from: {REAL_DIR}")
    X_real, y_real = load_data_ensemble(REAL_DIR, 0) # Label 0 for Real
    
    print(f"Loading FAKE data from: {FAKE_DIR}")
    X_fake, y_fake = load_data_ensemble(FAKE_DIR, 1) # Label 1 for Fake
    
    if X_real is None or X_fake is None:
        print("Error: Missing data. Please check your paths and run feature extraction first.")
        return

    # 2. 对齐时间步长
    min_len = min(X_real.shape[1], X_fake.shape[1])
    X_real = X_real[:, :min_len]
    X_fake = X_fake[:, :min_len]
    
    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([y_real, y_fake], axis=0)

    # 3. 数据标准化 (Standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\nDataset Summary:")
    print(f"  - Total Samples: {len(X)}")
    print(f"  - Feature Dimension: {X.shape[1]}")
    print(f"  - Training Set: {len(X_train)}")
    print(f"  - Test Set: {len(X_test)}")

    # 5. 定义模型群
    # 增加了正则化参数，因为特征维度增加了，防止过拟合
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=0.01, max_iter=1000, random_state=42)
    }

    # 6. 训练与评估循环
    print("\n" + "="*100)
    print(f"{'Model Name':<20} | {'Acc':<8} | {'AP':<8} | {'AUC':<8} | {'R_Acc':<8} | {'F_Acc':<8}")
    print("-" * 100)
    
    for name, clf in models.items():
        # 训练
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 计算概率 
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        
        # --- 指标计算 ---
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob) 
        
        # 计算 R_ACC (Real Accuracy) 和 F_ACC (Fake Accuracy)
        # label 0 = Real, label 1 = Fake
        r_acc = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
        f_acc = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
        
        print(f"{name:<20} | {acc:.4f}   | {ap:.4f}   | {auc:.4f}   | {r_acc:.4f}   | {f_acc:.4f}")

    print("="*100)

if __name__ == "__main__":
    main()