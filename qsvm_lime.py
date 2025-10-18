import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
import sys
import time
import numpy as np

# ====================================================================
# SIMPLIFIED QUANTUM GATES
# ====================================================================

def rx(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return qt.Qobj([[cos, -1j * sin], [-1j * sin, cos]])

def ry(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return qt.Qobj([[cos, -sin], [sin, cos]])

def rz(theta):
    return qt.Qobj([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])

def hadamard():
    return qt.Qobj([[1, 1], [1, -1]]) / np.sqrt(2)

# ====================================================================
# FAST QUANTUM FEATURE MAP (1-Layer, Minimal Entanglement)
# ====================================================================

def create_fast_quantum_state(x, n_qubits):
    """Fast 1-layer quantum feature map"""
    H = hadamard()
    state = qt.tensor([H * qt.basis(2, 0) for _ in range(n_qubits)])

    # Single layer with non-linear encoding
    for i in range(n_qubits):
        angle_rx = np.tanh(x[i] * 2.8) * np.pi
        angle_ry = np.sin(x[i] * 3.5) * np.pi
        
        Rx = rx(angle_rx)
        Ry = ry(angle_ry)
        
        op_rx = qt.tensor([Rx if j == i else qt.qeye(2) for j in range(n_qubits)])
        op_ry = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
        
        state = op_ry * op_rx * state

    return state.unit()

# ====================================================================
# FAST QUANTUM KERNEL (No Multiprocessing)
# ====================================================================

def compute_quantum_kernel_fast(X1, X2, n_qubits):
    """Fast kernel using direct overlap computation"""
    print(f"Computing quantum kernel for {len(X1)} x {len(X2)} samples...")
    
    states1 = [create_fast_quantum_state(x, n_qubits) for x in X1]
    states2 = [create_fast_quantum_state(x, n_qubits) for x in X2]

    kernel = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        bra = states1[i].dag()
        for j in range(len(X2)):
            overlap = bra * states2[j]
            kernel[i, j] = np.abs(overlap) ** 2
        if (i + 1) % max(1, len(X1) // 5) == 0:
            print(f"  Progress: {i + 1}/{len(X1)}")

    kernel /= np.max(kernel) + 1e-10
    if kernel.shape[0] == kernel.shape[1]:
        kernel += np.eye(kernel.shape[0]) * 1e-4

    return kernel

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 70)
    print("   🚀 OPTIMIZED QSVM - FAST VERSION")
    print("=" * 70)
    start_time = time.time()

    print("📁 Loading/Generating data...")
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
        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

        for col in ['protocol_type', 'service', 'flag']:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))

        train_df['label'] = (train_df['label'] != 'normal').astype(int)
        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values

        for col in numeric_cols:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        for col in ['protocol_type', 'service', 'flag']:
            le = LabelEncoder()
            test_df[col] = le.fit_transform(test_df[col].astype(str))

        test_df['label'] = (test_df['label'] != 'normal').astype(int)
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_components = 4
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        minmax = MinMaxScaler(feature_range=(0, np.pi / 2))
        X_train = minmax.fit_transform(X_train)
        X_test = minmax.transform(X_test)

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}. Generating dummy data.")
        n_components = 4
        X_train = np.random.rand(1000, n_components) * np.pi / 2
        y_train = np.random.randint(0, 2, 1000)
        X_test = np.random.rand(100, n_components) * np.pi / 2
        y_test = np.random.randint(0, 2, 100)

    # Aggressive sampling for speed
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=999, random_state=43, stratify=y_train)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=10, random_state=43, stratify=y_test)
    print(f"✅ Using {len(X_train)} training and {len(X_test)} test samples.\n")

    # --- CLASSICAL SVM BASELINE ---
    print("=" * 70)
    print("🔵 CLASSICAL SVM (LINEAR)")
    print("=" * 70)
    svm_classical = SVC(kernel='linear', C=10, random_state=43)
    svm_classical.fit(X_train, y_train)

    y_pred_classical = svm_classical.predict(X_test)
    accuracy_classical = svm_classical.score(X_test, y_test)
    precision_classical = precision_score(y_test, y_pred_classical, zero_division=0)
    recall_classical = recall_score(y_test, y_pred_classical, zero_division=0)
    f1_classical = f1_score(y_test, y_pred_classical, zero_division=0)

    print(f"📊 Results: Accuracy={accuracy_classical:.4f}, Precision={precision_classical:.4f}, Recall={recall_classical:.4f}\n")

    # --- QUANTUM SVM ---
    print("=" * 70)
    print("🔴 QUANTUM SVM (FAST)")
    print("=" * 70)
    n_qubits = n_components

    kernel_start = time.time()
    train_kernel = compute_quantum_kernel_fast(X_train, X_train, n_qubits)
    test_kernel = compute_quantum_kernel_fast(X_test, X_train, n_qubits)
    print(f"⏱️  Kernel time: {time.time() - kernel_start:.1f}s\n")

    svm_quantum = SVC(kernel='precomputed', C=100, random_state=43)
    svm_quantum.fit(train_kernel, y_train)

    y_pred_quantum = svm_quantum.predict(test_kernel)
    accuracy_quantum = svm_quantum.score(test_kernel, y_test)
    precision_quantum = precision_score(y_test, y_pred_quantum, zero_division=0)
    recall_quantum = recall_score(y_test, y_pred_quantum, zero_division=0)
    f1_quantum = f1_score(y_test, y_pred_quantum, zero_division=0)

    print(f"📊 Results: Accuracy={accuracy_quantum:.4f}, Precision={precision_quantum:.4f}, Recall={recall_quantum:.4f}\n")

    # --- COMPARISON ---
    print("=" * 70)
    print("🚀 QUANTUM ADVANTAGE")
    print(f"   Accuracy Δ: {(accuracy_quantum-accuracy_classical):+.4f}")
    print(f"   Precision Δ: {(precision_quantum-precision_classical):+.4f}")
    print(f"   Recall Δ: {(recall_quantum-recall_classical):+.4f}")
    print(f"   F1-Score Δ: {(f1_quantum-f1_classical):+.4f}")
    print("=" * 70)
    print(f"⏱️  Total runtime: {time.time() - start_time:.1f}s\n")

    # --- CONFUSION MATRIX ---
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred_quantum)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Reds')
    plt.title('QSVM Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()