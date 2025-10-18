import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
import time
from functools import lru_cache
import joblib
from joblib import Parallel, delayed
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.style as mplstyle
from sklearn.metrics import ConfusionMatrixDisplay

mplstyle.use('ggplot')

# ====================================================================
# QUANTUM GATES AND FEATURE MAP (Optimized for Speed)
# ====================================================================

@lru_cache(maxsize=512)
def rx(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return qt.Qobj([[cos, -1j * sin], [-1j * sin, cos]])

@lru_cache(maxsize=512)
def ry(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return qt.Qobj([[cos, -sin], [sin, cos]])

@lru_cache(maxsize=512)
def cnot_cached(N, control, target):
    P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    I = qt.qeye(2)
    X = qt.sigmax()
    ops0 = [qt.qeye(2)] * N
    ops0[control] = P0
    ops0[target] = I
    term0 = qt.tensor(ops0)
    ops1 = [qt.qeye(2)] * N
    ops1[control] = P1
    ops1[target] = X
    term1 = qt.tensor(ops1)
    return term0 + term1

def hadamard():
    return qt.Qobj([[1, 1], [1, -1]]) / np.sqrt(2)

@lru_cache(maxsize=1024)
def create_advanced_quantum_state(tuple_x, n_qubits):
    x = np.array(tuple_x)
    H = hadamard()
    state = qt.tensor([H * qt.basis(2, 0) for _ in range(n_qubits)])
    encoding_layers = [
        {'scale': 2.8, 'rx_w': 1.0, 'ry_w': 0.9},
        {'scale': 3.5, 'rx_w': 1.3, 'ry_w': 0.7},
    ]
    for layer_idx, layer_params in enumerate(encoding_layers):
        scale = layer_params['scale']
        for i in range(n_qubits):
            angle_base = x[i] * scale
            angle_rx = np.tanh(angle_base) * layer_params['rx_w'] * np.pi
            angle_ry = np.sin(angle_base) * layer_params['ry_w'] * np.pi
            Rx = rx(angle_rx)
            Ry = ry(angle_ry)
            op_rx = qt.tensor([Rx if j == i else qt.qeye(2) for j in range(n_qubits)])
            op_ry = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
            state = op_ry * op_rx * state
        for i in range(n_qubits - 1):
            cn = cnot_cached(n_qubits, i, i + 1)
            state = cn * state
    return state.unit()

def compute_row(states1, states2, i):
    row = np.zeros(len(states2))
    bra = states1[i].dag()
    for j in range(len(states2)):
        overlap = bra * states2[j]
        row[j] = np.abs(overlap) ** 2
    return row

def compute_quantum_kernel_advanced(X1, X2, n_qubits, batch_size=100):
    print(f"Computing quantum kernel for {len(X1)} x {len(X2)} samples...")
    states1 = [create_advanced_quantum_state(tuple(x), n_qubits) for x in X1]
    states2 = states1 if np.array_equal(X1, X2) else [create_advanced_quantum_state(tuple(x), n_qubits) for x in X2]
    print("  Computing quantum fidelity matrix...")
    kernel = np.array(Parallel(n_jobs=-1)(delayed(compute_row)(states1, states2, i) for i in range(len(X1))))
    kernel /= np.max(kernel) + 1e-10
    if kernel.shape[0] == kernel.shape[1]:
        kernel += np.eye(kernel.shape[0]) * 1e-4
    return kernel

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 50)
    print("🚀 OPTIMIZED QSVM")
    print("=" * 50)
    qsvm_start = time.time()

    print("📁 Loading NSL-KDD dataset...")
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    try:
        train_df = pd.read_csv('KDDTrain+.txt', header=None)
        test_df = pd.read_csv('KDDTest+.txt', header=None)
        train_df.columns = columns + ['label']
        test_df.columns = columns + ['label']

        numeric_cols = [col for col in train_df.columns if col not in ['protocol_type', 'service', 'flag', 'label']]
        categorical_cols = ['protocol_type', 'service', 'flag']
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            le_dict[col] = le
        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

        train_df['label'] = (train_df['label'] != 'normal').astype(int)
        test_df['label'] = (test_df['label'] != 'normal').astype(int)

        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_components = 3
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)

        minmax = MinMaxScaler(feature_range=(0, np.pi / 2))
        X_train_reduced = minmax.fit_transform(X_train_reduced)
        X_test_reduced = minmax.transform(X_test_reduced)

        train_size = 2000
        test_size = 5
        X_train_reduced, _, y_train, _ = train_test_split(
            X_train_reduced, y_train, train_size=train_size, random_state=42, stratify=y_train
        )
        X_test_reduced, _, y_test, _ = train_test_split(
            X_test_reduced, y_test, train_size=test_size, random_state=42, stratify=y_test
        )
        print(f"✅ Selected {train_size} training and {test_size} test samples.\n")

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}. Generating dummy data.")
        n_components = 3
        train_size = 2000
        test_size = 5
        X_train_reduced = np.random.rand(train_size, n_components) * np.pi/2
        y_train = np.random.randint(0, 2, train_size)
        X_test_reduced = np.random.rand(test_size, n_components) * np.pi/2
        y_test = np.random.randint(0, 2, test_size)
        print(f"✅ Generated {train_size} training and {test_size} test samples.\n")

    # --- CLASSICAL SVM ---
    print("=" * 50)
    print("🔵 CLASSICAL SVM")
    print("=" * 50)
    svm_classical = SVC(kernel='linear', C=0.01, random_state=42)
    svm_classical.fit(X_train_reduced, y_train)
    y_pred_classical = svm_classical.predict(X_test_reduced)
    accuracy_classical = svm_classical.score(X_test_reduced, y_test)
    precision_classical = precision_score(y_test, y_pred_classical, zero_division=0)
    recall_classical = recall_score(y_test, y_pred_classical, zero_division=0)
    f1_classical = f1_score(y_test, y_pred_classical, zero_division=0)
    y_score_classical = svm_classical.decision_function(X_test_reduced)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, y_score_classical)
    roc_auc_classical = auc(fpr_classical, tpr_classical)
    print(f"📊 Classical: Acc={accuracy_classical:.4f}, Prec={precision_classical:.4f}, "
          f"Rec={recall_classical:.4f}, F1={f1_classical:.4f}, AUC={roc_auc_classical:.4f}\n")

    # --- QUANTUM SVM ---
    print("=" * 50)
    print("🔴 QUANTUM SVM")
    print("=" * 50)
    n_qubits = n_components
    kernel_start = time.time()
    train_kernel = compute_quantum_kernel_advanced(X_train_reduced, X_train_reduced, n_qubits)
    test_kernel = compute_quantum_kernel_advanced(X_test_reduced, X_train_reduced, n_qubits)
    kernel_time = time.time() - kernel_start
    print(f"⏱️ Kernel time: {kernel_time:.1f}s")

    param_grid = {'C': [1, 100, 1000]}
    grid = GridSearchCV(SVC(kernel='precomputed', probability=True), param_grid, cv=3, scoring='f1')
    grid.fit(train_kernel, y_train)
    svm_quantum = grid.best_estimator_
    print(f"✅ Optimal C: {grid.best_params_['C']}")

    y_pred_quantum = svm_quantum.predict(test_kernel)
    accuracy_quantum = svm_quantum.score(test_kernel, y_test)
    precision_quantum = precision_score(y_test, y_pred_quantum, zero_division=0)
    recall_quantum = recall_score(y_test, y_pred_quantum, zero_division=0)
    f1_quantum = f1_score(y_test, y_pred_quantum, zero_division=0)
    y_score_quantum = svm_quantum.decision_function(test_kernel)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_score_quantum)
    roc_auc_quantum = auc(fpr_quantum, tpr_quantum)

    print(f"📊 Quantum: Acc={accuracy_quantum:.4f} (Δ {(accuracy_quantum-accuracy_classical):+.4f})")
    print(f"   Prec={precision_quantum:.4f} (Δ {(precision_quantum-precision_classical):+.4f})")
    print(f"   Rec={recall_quantum:.4f} (Δ {(recall_quantum-recall_classical):+.4f})")
    print(f"   F1={f1_quantum:.4f} (Δ {(f1_quantum-f1_classical):+.4f})")
    print(f"   AUC={roc_auc_quantum:.4f} (Δ {(roc_auc_quantum-roc_auc_classical):+.4f})")
    print(f"⏱️ Total time: {time.time() - qsvm_start:.1f}s\n")

    # --- VISUALIZATIONS ---
    print("=" * 50)
    print("📈 VISUALIZATIONS")
    print("=" * 50)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr_classical, tpr_classical, label=f'Classical (AUC = {roc_auc_classical:.2f})')
    ax1.plot(fpr_quantum, tpr_quantum, label=f'Quantum (AUC = {roc_auc_quantum:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_quantum)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=ax2, cmap='Reds')
    ax2.set_title('QSVM Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # --- LIME ---
    print("=" * 50)
    print("🧠 LIME EXPLANATION")
    print("=" * 50)
    def qsvm_predict_proba(X):
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        K = compute_quantum_kernel_advanced(X, X_train_reduced, n_qubits)
        return svm_quantum.predict_proba(K)

    feature_names = [f'PC{i+1}' for i in range(X_train_reduced.shape[1])]
    explainer = LimeTabularExplainer(
        X_train_reduced, feature_names=feature_names, class_names=['Normal', 'Anomaly'], mode='classification'
    )
    instance_idx = 0
    instance = X_test_reduced[instance_idx]
    exp = explainer.explain_instance(instance, qsvm_predict_proba, num_features=3, num_samples=200)
    print(f"Explaining instance {instance_idx}: True={y_test[instance_idx]}, Pred={y_pred_quantum[instance_idx]}")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")

if __name__ == '__main__':
    main()