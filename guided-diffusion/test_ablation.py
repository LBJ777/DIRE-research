"""
test_ablation.py
"""
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

REAL_DIR = "../results/LSUN/real"
FAKE_DIR = "../results/LSUN/fake"

def load_data():
    print("Loading data...")
    # 加载所有数据
    real_files = glob.glob(os.path.join(REAL_DIR, "*.npy"))
    fake_files = glob.glob(os.path.join(FAKE_DIR, "*.npy"))
    
    X_list = []
    y_list = []
    
    for f in real_files:
        try: 
            arr = np.load(f)
            if arr.shape[1] >= 5: X_list.append(arr); y_list.append(0)
        except: pass
        
    for f in fake_files:
        try:
            arr = np.load(f)
            if arr.shape[1] >= 5: X_list.append(arr); y_list.append(1)
        except: pass
        
    # [N, T, 5] -> Mean, Min, Std, Skew, Kurt
    X_raw = np.array(X_list) 
    y = np.array(y_list)
    
    # 展平特征 [N, T*5]
    # 我们需要手动切分特征
    # Consistency Features: Indices 0, 1, 2 (Mean, Min, Std)
    # Gaussianity Features: Indices 3, 4 (Skew, Kurt)
    
    T = X_raw.shape[1]
    
    # 1. Consistency Only (前3列)
    X_cons = X_raw[:, :, 0:3].reshape(len(X_raw), -1)
    
    # 2. Gaussianity Only (后2列)
    X_gauss = X_raw[:, :, 3:5].reshape(len(X_raw), -1)
    
    # 3. Fusion (全5列)
    X_fusion = X_raw.reshape(len(X_raw), -1)
    
    return X_cons, X_gauss, X_fusion, y

def train_eval(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # --- 指标计算 ---
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    
    r_acc = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
    f_acc = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
    
    print(f"{name:<20} | Acc: {acc:.4f} | AP: {ap:.4f} | AUC: {auc:.4f} | R_Acc: {r_acc:.4f} | F_Acc: {f_acc:.4f}")

def main():
    X_cons, X_gauss, X_fusion, y = load_data()
    
    print("\n" + "="*85)
    print("Ablation Study Results")
    print("="*85)
    
    train_eval(X_cons, y,   "Only Consistency")
    train_eval(X_gauss, y,  "Only Gaussianity")
    train_eval(X_fusion, y, "Fusion (Both)")
    print("="*85)

if __name__ == "__main__":
    main()