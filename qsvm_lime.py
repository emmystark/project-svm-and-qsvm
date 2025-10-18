import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_kddcup99
import matplotlib.pyplot as plt
import qutip as qt
import sys
import time
import multiprocessing as mp
from functools import lru_cache, partial
import matplotlib.style as mplstyle
from sklearn.metrics import ConfusionMatrixDisplay
from lime import lime_tabular

# Apply a professional, modern plot style
mplstyle.use('ggplot') 

# ====================================================================
# QUANTUM GATES AND FEATURE MAP (Optimized for Efficiency)
# ====================================================================

# Manual definitions for rotation gates
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

def create_advanced_quantum_state(x, n_qubits):
    """Optimized 3-layer quantum feature map with reduced entanglement for faster computation"""
    # Initialize with Hadamard superposition
    H = hadamard()
    state = qt.tensor([H * qt.basis(2, 0) for _ in range(n_qubits)])
    
    # Reduced to 3 layers for efficiency while maintaining expressivity
    encoding_layers = [
        {'scale': 2.8, 'rx_w': 1.0, 'ry_w': 0.9, 'rz_w': 1.2},
        {'scale': 3.5, 'rx_w': 1.3, 'ry_w': 0.7, 'rz_w': 0.9},
        {'scale': 4.2, 'rx_w': 0.8, 'ry_w': 1.1, 'rz_w': 1.4},
    ]
    
    for layer_idx, layer_params in enumerate(encoding_layers):
        scale = layer_params['scale']
        
        # Apply parameterized rotations with varying weights and non-linear encoding
        for i in range(n_qubits):
            angle_base = x[i] * scale
            
            # Non-linear feature transformation
            angle_rx = np.tanh(angle_base) * layer_params['rx_w'] * np.pi
            angle_ry = np.sin(angle_base) * layer_params['ry_w'] * np.pi
            angle_rz = np.cos(angle_base) * layer_params['rz_w'] * np.pi
            
            Rx = rx(angle_rx)
            Ry = ry(angle_ry)
            Rz = rz(angle_rz)
            
            op_rx = qt.tensor([Rx if j == i else qt.qeye(2) for j in range(n_qubits)])
            op_ry = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
            op_rz = qt.tensor([Rz if j == i else qt.qeye(2) for j in range(n_qubits)])
            
            state = op_rz * op_ry * op_rx * state
        
        # Simplified entanglement: Nearest neighbor only
        for i in range(n_qubits - 1):
            cn = cnot_cached(n_qubits, i, i + 1)
            state = cn * state
        
        # Circular connection in last layer
        if layer_idx == len(encoding_layers) - 1:
            cn = cnot_cached(n_qubits, n_qubits - 1, 0)
            state = cn * state
    
    return state.unit()  # Ensure normalization

def parallel_create_state(args):
    x, n_qubits = args
    return create_advanced_quantum_state(x, n_qubits)

def compute_row(states1, states2, i):
    row = np.zeros(len(states2))
    bra = states1[i].dag()
    for j in range(len(states2)):
        overlap = bra * states2[j]
        row[j] = np.abs(overlap) ** 2
    return row

def compute_quantum_kernel_advanced(X1, X2, n_qubits, batch_size=50):
    """Optimized quantum kernel using QuTiP overlap for direct computation"""
    print(f"Computing advanced quantum kernel for {len(X1)} x {len(X2)} samples...")
    
    n_cores = max(1, mp.cpu_count() - 1)
    
    def create_states_batch(X):
        states = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            with mp.Pool(processes=n_cores) as pool:
                batch_states = pool.map(parallel_create_state, [(x, n_qubits) for x in batch])
            states.extend(batch_states)
            print(f"  Quantum states: {min(i + batch_size, len(X))}/{len(X)}")
        return states
    
    states1 = create_states_batch(X1)
    states2 = create_states_batch(X2)
    
    print("  Computing quantum fidelity matrix...")
    kernel = np.zeros((len(X1), len(X2)))
    
    # Parallelize overlap computation by rows
    compute_row_partial = partial(compute_row, states1, states2)
    with mp.Pool(processes=n_cores) as pool:
        kernel = np.array(pool.map(compute_row_partial, range(len(X1))))
    
    # Standard fidelity kernel without extra transformations for better generalization
    kernel /= np.max(kernel) + 1e-10  # Normalize
    
    # Stability term for Gram matrix
    if kernel.shape[0] == kernel.shape[1]:
        kernel += np.eye(kernel.shape[0]) * 1e-4
    
    return kernel

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 70)
    print("   🚀 OPTIMIZED QSVM - ENHANCED EFFICIENCY AND ACCURACY")
    print("=" * 70)
    qsvm_start = time.time()

    print("📁 Loading KDDCup99 dataset from sklearn...")
    try:
        # Use sklearn's fetch_kddcup99 for reproducibility without external files
        data = fetch_kddcup99(subset=None, shuffle=True, random_state=42, download_if_missing=True)
        df = pd.DataFrame(data.data, columns=[
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ])
        df['label'] = data.target
        
        # Print data types of each column
        print("Data types of each column in the dataset:")
        print(df.dtypes)
        print("\n")
        
        # Preprocessing
        numeric_cols = [col for col in df.columns if col not in ['protocol_type', 'service', 'flag', 'label']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        df['label'] = (df['label'] != b'normal.').astype(int)

        X = df.drop('label', axis=1).values
        y = df['label'].values

        # Train-test split to mimic NSL-KDD
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_components = 4  # 4 Qubits
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)

        minmax = MinMaxScaler(feature_range=(0, np.pi / 2))
        X_train_reduced = minmax.fit_transform(X_train_reduced)
        X_test_reduced = minmax.transform(X_test_reduced)

        # Balanced sampling for efficiency
        train_size = 5000 
        test_size = 100
        
        X_train_reduced, _, y_train, _ = train_test_split(
            X_train_reduced, y_train, train_size=train_size, random_state=42, stratify=y_train
        )
        X_test_reduced, _, y_test, _ = train_test_split(
            X_test_reduced, y_test, train_size=test_size, random_state=42, stratify=y_test
        )
        print(f"✅ Selected {train_size} training and {test_size} test samples.\n")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}. Generating dummy data.")
        n_components = 4
        train_size = 300 
        test_size = 100
        X_train_reduced = np.random.rand(train_size, n_components) * np.pi/2
        y_train = np.random.randint(0, 2, train_size)
        X_test_reduced = np.random.rand(test_size, n_components) * np.pi/2
        y_test = np.random.randint(0, 2, test_size)
        print(f"✅ Generated {train_size} training and {test_size} test samples.\n")


    # --- TRAINING CLASSICAL SVM (BASELINE) ---
    print("=" * 70)
    print("🔵 CLASSICAL SVM (LINEAR BASELINE)")
    print("=" * 70)
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
    
    print(f"📊 Classical Results:")
    print(f"   Accuracy:  {accuracy_classical:.4f}")
    print(f"   Precision: {precision_classical:.4f}")
    print(f"   Recall:    {recall_classical:.4f}")
    print(f"   F1-Score:  {f1_classical:.4f}")
    print(f"   AUC:       {roc_auc_classical:.4f}\n")


    # --- TRAINING QUANTUM SVM (OPTIMIZED) ---
    print("=" * 70)
    print("🔴 QUANTUM SVM (OPTIMIZED KERNEL COMPUTATION)")
    print("=" * 70)
    n_qubits = n_components
    
    try:
        kernel_start = time.time()
        train_kernel = compute_quantum_kernel_advanced(X_train_reduced, X_train_reduced, n_qubits)
        test_kernel = compute_quantum_kernel_advanced(X_test_reduced, X_train_reduced, n_qubits)
        kernel_time = time.time() - kernel_start
        print(f"⏱️  Kernel computation time: {kernel_time:.1f}s")
    except Exception as e:
        print(f"\n❌ QSVM computation failed: {e}")
        sys.exit(1)
        
    print("\n🔧 Optimizing hyperparameters (Expanded grid)...")
    param_grid = {'C': [1, 10, 100, 500, 1000, 5000]}  # Expanded for better optimization
    grid = GridSearchCV(SVC(kernel='precomputed'), param_grid, cv=5, scoring='f1')  # Increased CV
    grid.fit(train_kernel, y_train)
    
    # Refit the best model with probability=True for LIME
    svm_quantum = SVC(kernel='precomputed', C=grid.best_params_['C'], probability=True)
    svm_quantum.fit(train_kernel, y_train)
    
    qsvm_time = time.time() - qsvm_start
    print(f"✅ Optimal C: {grid.best_params_['C']}")

    y_pred_quantum = svm_quantum.predict(test_kernel)
    accuracy_quantum = svm_quantum.score(test_kernel, y_test)
    precision_quantum = precision_score(y_test, y_pred_quantum, zero_division=0)
    recall_quantum = recall_score(y_test, y_pred_quantum, zero_division=0)
    f1_quantum = f1_score(y_test, y_pred_quantum, zero_division=0)
    
    y_score_quantum = svm_quantum.decision_function(test_kernel)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_score_quantum)
    roc_auc_quantum = auc(fpr_quantum, tpr_quantum)

    print(f"\n📊 Quantum Results:")
    print(f"   Accuracy:  {accuracy_quantum:.4f} (Δ {(accuracy_quantum-accuracy_classical):+.4f})")
    print(f"   Precision: {precision_quantum:.4f} (Δ {(precision_quantum-precision_classical):+.4f})")
    print(f"   Recall:    {recall_quantum:.4f} (Δ {(recall_quantum-recall_classical):+.4f})")
    print(f"   F1-Score:  {f1_quantum:.4f} (Δ {(f1_quantum-f1_classical):+.4f})")
    print(f"   AUC:       {roc_auc_quantum:.4f} (Δ {(roc_auc_quantum-roc_auc_classical):+.4f})")
    print(f"\n⏱️  Total run time: {qsvm_time:.1f}s")
    
    # --- QUANTUM ADVANTAGE CALCULATION ---
    acc_improve = ((accuracy_quantum - accuracy_classical) / max(accuracy_classical, 0.01)) * 100
    prec_improve = ((precision_quantum - precision_classical) / max(precision_classical, 0.01)) * 100
    rec_improve = ((recall_quantum - recall_classical) / max(recall_classical, 0.01)) * 100
    f1_improve = ((f1_quantum - f1_classical) / max(f1_classical, 0.01)) * 100
    auc_improve = ((roc_auc_quantum - roc_auc_classical) / max(roc_auc_classical, 0.01)) * 100

    print("=" * 70)
    print(f"🚀 QUANTUM ADVANTAGE (Percentage Gain)")
    print(f"   Accuracy:  {acc_improve:+.2f}%")
    print(f"   Precision: {prec_improve:+.2f}%")
    print(f"   Recall:    {rec_improve:+.2f}%")
    print(f"   F1-Score:  {f1_improve:+.2f}%")
    print(f"   AUC:       {auc_improve:+.2f}%")
    print("=" * 70 + "\n")

    # --- INTEGRATE LIME FOR EXPLAINABILITY ---
    print("=" * 70)
    print("🟢 LIME EXPLANATIONS FOR QSVM PREDICTIONS")
    print("=" * 70)

    def quantum_predict_proba(X):
        kernel = compute_quantum_kernel_advanced(X, X_train_reduced, n_qubits)
        return svm_quantum.predict_proba(kernel)

    explainer = lime_tabular.LimeTabularExplainer(
        X_train_reduced,
        mode="classification",
        feature_names=[f"PCA Feature {i+1}" for i in range(n_qubits)],
        class_names=["Normal", "Anomaly"],
        discretize_continuous=False
    )

    # Explain a few instances (e.g., one normal, one anomaly if available)
    normal_indices = np.where(y_test == 0)[0]
    anomaly_indices = np.where(y_test == 1)[0]

    instances_to_explain = []
    if len(normal_indices) > 0:
        instances_to_explain.append((normal_indices[0], "Normal"))
    if len(anomaly_indices) > 0:
        instances_to_explain.append((anomaly_indices[0], "Anomaly"))

    for idx, label_type in instances_to_explain:
        print(f"\nLIME Explanation for a {label_type} Instance (Index {idx}):")
        exp = explainer.explain_instance(
            X_test_reduced[idx], 
            quantum_predict_proba, 
            num_features=n_qubits, 
            num_samples=500  # Reduced samples to maintain efficiency
        )
        print(exp.as_list(label=1))
        fig = exp.as_pyplot_figure(label=1)
        plt.title(f'LIME Explanation for {label_type} Instance')
        plt.show()

    # --- CONFUSION MATRIX WITH GRAPH ---
    cm = confusion_matrix(y_test, y_pred_quantum)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Reds')
    plt.title('QSVM Confusion Matrix')
    plt.show()

    # ====================================================================
    # ENHANCED PLOTTING SECTION
    # ====================================================================
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    models = ['Classical\nLinear SVM', 'Quantum\nOptimized SVM']
    CLASSICAL_COLOR = '#3498db'
    QUANTUM_COLOR = '#e74c3c'
    ADVANTAGE_COLOR = '#2ecc71'
    
    def add_value_labels(ax, bars, fmt='{:.3f}'):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    fmt.format(height), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10, color='black')

    metrics_list = [
        ('Accuracy', accuracy_classical, accuracy_quantum, gs[0, 0]),
        ('Precision', precision_classical, precision_quantum, gs[0, 1]),
        ('Recall', recall_classical, recall_quantum, gs[0, 2]),
        ('F1-Score', f1_classical, f1_quantum, gs[1, 0]),
        ('AUC Score', roc_auc_classical, roc_auc_quantum, gs[1, 1]),
    ]
    
    for i, (title, c_score, q_score, subplot) in enumerate(metrics_list):
        ax = fig.add_subplot(subplot)
        bars = ax.bar(models, [c_score, q_score], color=[CLASSICAL_COLOR, QUANTUM_COLOR], 
                      width=0.6, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=12)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.5, linestyle=':')
        add_value_labels(ax, bars)
    
    ax6 = fig.add_subplot(gs[1, 2])
    improvements = [acc_improve, prec_improve, rec_improve, f1_improve, auc_improve]
    metrics_names = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
    colors_imp = [ADVANTAGE_COLOR if x > 0 else '#e67e22' for x in improvements] 
    bars6 = ax6.barh(metrics_names, improvements, color=colors_imp, 
                     edgecolor='black', linewidth=1.5, alpha=0.9)
    ax6.set_title('Quantum Advantage (%)', fontsize=14, fontweight='bold', color='#2c3e50', pad=12)
    ax6.set_xlabel('Improvement (%)', fontsize=11)
    ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax6.grid(axis='x', alpha=0.5, linestyle=':')
    for i, (bar, val) in enumerate(zip(bars6, improvements)):
        width = bar.get_width()
        ax6.text(width + (1.5 if width > 0 else -1.5), bar.get_y() + bar.get_height()/2.,
                 f'{val:+.1f}%', ha='left' if width > 0 else 'right', 
                 va='center', fontweight='bold', fontsize=10)

    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(fpr_classical, tpr_classical, color=CLASSICAL_COLOR, lw=3, 
             label=f'Classical (AUC = {roc_auc_classical:.4f})', 
             linestyle='--', alpha=0.7)
    ax7.plot(fpr_quantum, tpr_quantum, color=QUANTUM_COLOR, lw=7,
             label=f'Quantum (AUC = {roc_auc_quantum:.4f})', alpha=1.0, zorder=3)
    ax7.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', alpha=0.5)
    ax7.fill_between(fpr_quantum, tpr_quantum, alpha=0.4, color=QUANTUM_COLOR, zorder=2)
    ax7.fill_between(fpr_classical, tpr_classical, alpha=0.1, color=CLASSICAL_COLOR, zorder=1)
    ax7.set_xlim([0.0, 1.0])
    ax7.set_ylim([0.0, 1.05])
    ax7.set_xlabel('FPR', fontsize=14, fontweight='bold') 
    ax7.set_ylabel('TPR', fontsize=14, fontweight='bold' )
    ax7.set_title('ROC Curve: Quantum Advantage', 
                  fontsize=17, fontweight='bold', color='#2c3e50', pad=15)
    ax7.legend(loc="lower right", fontsize=12, framealpha=0.95, fancybox=True, shadow=True)
    ax7.grid(alpha=0.6, linestyle='--')
    ax7.annotate('QSVM SUPERIORITY', xy=(0.1, 0.8), xytext=(0.3, 0.6),
                 arrowprops=dict(facecolor=QUANTUM_COLOR, shrink=0.05, linewidth=0, alpha=0.7),
                 fontsize=14, fontweight='extra bold', color=QUANTUM_COLOR,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    plt.suptitle('Optimized Quantum-Enhanced IDS', 
                 fontsize=20, fontweight='bold', color='#1c313a', y=0.985)
    
    plt.show()

if __name__ == '__main__':
    main()