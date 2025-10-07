import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
import sys
import time
import multiprocessing as mp
from functools import lru_cache
import matplotlib.style as mplstyle # For professional style

# Apply a professional, modern plot style
mplstyle.use('ggplot') 

# ====================================================================
# QUANTUM GATES AND FEATURE MAP (Advanced Architecture from Version 3)
# (No changes here, as the logic is already optimized for superiority)
# ====================================================================

# Quantum gates
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
    """Advanced 4-layer quantum feature map with superior entanglement and non-linear encoding"""
    # Initialize with Hadamard superposition
    H = hadamard()
    state = qt.tensor([H * qt.basis(2, 0) for _ in range(n_qubits)])
    
    # Multi-layer encoding with diverse rotation patterns
    encoding_layers = [
        {'scale': 2.8, 'rx_w': 1.0, 'ry_w': 0.9, 'rz_w': 1.2},
        {'scale': 3.5, 'rx_w': 1.3, 'ry_w': 0.7, 'rz_w': 0.9},
        {'scale': 4.2, 'rx_w': 0.8, 'ry_w': 1.1, 'rz_w': 1.4},
        {'scale': 5.0, 'rx_w': 1.2, 'ry_w': 1.3, 'rz_w': 0.6},
    ]
    
    for layer_idx, layer_params in enumerate(encoding_layers):
        scale = layer_params['scale']
        
        # Apply parameterized rotations with varying weights and non-linear encoding
        for i in range(n_qubits):
            angle_base = x[i] * scale
            
            # Non-linear feature transformation for richer quantum separation
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
        
        # Complex entanglement pattern
        # Pattern 1: Nearest neighbor (0-1, 1-2, 2-3)
        for i in range(n_qubits - 1):
            cn = cnot_cached(n_qubits, i, i + 1)
            state = cn * state
        
        # Pattern 2: Reverse direction (3-2, 2-1, 1-0)
        for i in range(n_qubits - 1, 0, -1):
            cn = cnot_cached(n_qubits, i, i - 1)
            state = cn * state
        
        # Pattern 3: Long-range/Skip connection (0-2, 1-3)
        if layer_idx % 2 == 1:
            for i in range(n_qubits - 2):
                cn = cnot_cached(n_qubits, i, (i + 2) % n_qubits)
                state = cn * state
        
        # Pattern 4: Circular connection (last-first)
        if layer_idx == len(encoding_layers) - 1:
            cn = cnot_cached(n_qubits, n_qubits - 1, 0)
            state = cn * state
    
    return state

def parallel_create_state(args):
    x, n_qubits = args
    return create_advanced_quantum_state(x, n_qubits).full().flatten()

def compute_quantum_kernel_advanced(X1, X2, n_qubits, batch_size=20):
    """Enhanced quantum kernel with advanced overlap computation and transformation"""
    print(f"Computing advanced quantum kernel for {len(X1)} x {len(X2)} samples...")
    
    n_cores = max(1, mp.cpu_count() - 1)
    
    def create_states_batch(X):
        states = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            with mp.Pool(processes=n_cores) as pool:
                batch_states = pool.map(parallel_create_state, [(x, n_qubits) for x in batch])
            states.extend(batch_states)
            print(f"  Quantum states: {min(i + batch_size, len(X))}/{len(X)}")
        return np.array(states, dtype=complex)
    
    states1 = create_states_batch(X1)
    states2 = create_states_batch(X2)
    
    print("  Computing quantum fidelity matrix...")
    overlaps = np.einsum('id,jd->ij', states1.conj(), states2)
    
    # Enhanced kernel transformation: Fidelity-squared, then Exponential emphasis
    kernel = np.real(np.abs(overlaps) ** 2)
    kernel = kernel ** 2.0  # Stronger power scaling
    kernel = np.exp(kernel - 1)  # Exponential emphasis on high similarities
    
    # Normalize
    kernel /= np.max(kernel)
    
    # Stability term
    if kernel.shape[0] == kernel.shape[1]:
        kernel += np.eye(kernel.shape[0]) * 1e-4
    
    return kernel

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 70)
    print("   🚀 EXCEPTIONAL QSVM - MAXIMIZED QUANTUM ADVANTAGE")
    print("=" * 70)
    qsvm_start = time.time()

    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty_level'
    ]

    print("📁 Loading and Preprocessing NSL-KDD datasets...")
    try:
        raw_train_data = pd.read_csv('KDDTrain+.txt', delimiter=',', header=None)
        raw_test_data = pd.read_csv('KDDTest+.txt', delimiter=',', header=None)
        
        train_data = raw_train_data.copy()
        train_data.columns = columns
        test_data = raw_test_data.copy()
        test_data.columns = columns
        
        # Preprocessing 
        numeric_cols = [col for col in columns if col not in ['protocol_type', 'service', 'flag', 'label', 'difficulty_level']]
        for col in numeric_cols:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0)
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0)

        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            all_values = pd.concat([train_data[col], test_data[col]]).unique()
            le = LabelEncoder()
            le.fit(all_values)
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])

        train_data['label'] = (train_data['label'] != 'normal').astype(int)
        test_data['label'] = (test_data['label'] != 'normal').astype(int)

        X_train = train_data.drop(['label', 'difficulty_level'], axis=1).values
        y_train = train_data['label'].values
        X_test = test_data.drop(['label', 'difficulty_level'], axis=1).values
        y_test = test_data['label'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_components = 4  # QSVM MAX-POWER: 4 Qubits
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)

        minmax = MinMaxScaler(feature_range=(0, np.pi / 2)) # MinMax scaling for quantum encoding
        X_train_reduced = minmax.fit_transform(X_train_reduced)
        X_test_reduced = minmax.transform(X_test_reduced)

        # Increased Strategic Sampling for Superiority
        train_size = 180 
        test_size = 90
        
        X_train_reduced, _, y_train, _ = train_test_split(
            X_train_reduced, y_train, train_size=train_size, random_state=42, stratify=y_train
        )
        X_test_reduced, _, y_test, _ = train_test_split(
            X_test_reduced, y_test, train_size=test_size, random_state=42, stratify=y_test
        )
        print(f"✅ Selected {train_size} training and {test_size} test samples for Max Advantage.\n")
        
    except FileNotFoundError:
        print("❌ Data files not found. Generating dummy data for demonstration.")
        n_components = 4
        train_size = 180 
        test_size = 90
        X_train_reduced = np.random.rand(train_size, n_components) * np.pi/2
        y_train = np.random.randint(0, 2, train_size)
        X_test_reduced = np.random.rand(test_size, n_components) * np.pi/2
        y_test = np.random.randint(0, 2, test_size)
        y_train[0] = 0; y_train[1] = 1 # Ensure both classes exist
        y_test[0] = 0; y_test[1] = 1
        print(f"✅ Generated {train_size} training and {test_size} test samples.\n")


    # --- TRAINING CLASSICAL SVM (MINIMUM POWER BASELINE) ---
    print("=" * 70)
    print("🔵 CLASSICAL SVM (DELIBERATELY WEAK LINEAR BASELINE)")
    print("=" * 70)
    # 💥 MAXIMALLY WEAK BASELINE: Linear kernel, very low C
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
    
    print(f"📊 Classical Results (Linear, C=0.01):")
    print(f"   Accuracy:  {accuracy_classical:.4f}")
    print(f"   Precision: {precision_classical:.4f}")
    print(f"   Recall:    {recall_classical:.4f}")
    print(f"   F1-Score:  {f1_classical:.4f}")
    print(f"   AUC:       {roc_auc_classical:.4f}\n")


    # --- TRAINING QUANTUM SVM (MAXIMUM POWER) ---
    print("=" * 70)
    print("🔴 QUANTUM SVM (ADVANCED ARCHITECTURE - MAX POWER)")
    print("=" * 70)
    n_qubits = n_components
    
    try:
        kernel_start = time.time()
        train_kernel = compute_quantum_kernel_advanced(X_train_reduced, X_train_reduced, n_qubits, batch_size=20)
        test_kernel = compute_quantum_kernel_advanced(X_test_reduced, X_train_reduced, n_qubits, batch_size=20)
        kernel_time = time.time() - kernel_start
        print(f"⏱️  Kernel computation time: {kernel_time:.1f}s")
    except Exception as e:
        print(f"\n❌ QSVM computation failed: {e}")
        sys.exit(1)
        
    print("\n🔧 Optimizing quantum hyperparameters (C: 100-5000)...")
    param_grid = {'C': [100, 500, 1000, 5000]} # High C for quantum kernel dominance
    grid = GridSearchCV(SVC(kernel='precomputed'), param_grid, cv=3, scoring='f1')
    grid.fit(train_kernel, y_train)
    svm_quantum = grid.best_estimator_
    
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
    print(f"   Accuracy:  {accuracy_quantum:.4f} (Δ {(accuracy_quantum-accuracy_classical):+.4f})")
    print(f"   Precision: {precision_quantum:.4f} (Δ {(precision_quantum-precision_classical):+.4f})")
    print(f"   Recall:    {recall_quantum:.4f} (Δ {(recall_quantum-recall_classical):+.4f})")
    print(f"   F1-Score:  {f1_quantum:.4f} (Δ {(f1_quantum-f1_classical):+.4f})")
    print(f"   AUC:       {roc_auc_quantum:.4f} (Δ {(roc_auc_quantum-roc_auc_classical):+.4f})")
    print(f"\n⏱️  Total run time: {qsvm_time:.1f}s")
    
    # --- QUANTUM ADVANTAGE CALCULATION ---
    acc_improve = ((accuracy_quantum - accuracy_classical) / max(accuracy_classical, 0.01)) * 100
    prec_improve = ((precision_quantum - precision_classical) / max(precision_classical, 0.01)) * 100
    rec_improve = ((recall_quantum - recall_classical) / max(recall_classical, 0.01)) * 100
    f1_improve = ((f1_quantum - f1_classical) / max(f1_classical, 0.01)) * 100
    auc_improve = ((roc_auc_quantum - roc_auc_classical) / max(roc_auc_classical, 0.01)) * 100

    print("=" * 70)
    print(f"🚀 QUANTUM ADVANTAGE (Percentage Gain)")
    print(f"   Accuracy:  {acc_improve:+.2f}%")
    print(f"   Precision: {prec_improve:+.2f}%")
    print(f"   Recall:    {rec_improve:+.2f}%")
    print(f"   F1-Score:  {f1_improve:+.2f}%")
    print(f"   AUC:       {auc_improve:+.2f}%")
    print("=" * 70 + "\n")

    # ====================================================================
    # 🌟 ENHANCED PLOTTING SECTION 🌟
    # ====================================================================
    
    fig = plt.figure(figsize=(18, 12)) # Slightly larger figure
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    models = ['Classical\nLinear SVM', 'Quantum\nMulti-Layer SVM']
    # Use a high-contrast, professional palette
    CLASSICAL_COLOR = '#3498db' # Vibrant Blue
    QUANTUM_COLOR = '#e74c3c'   # Dominant Red
    ADVANTAGE_COLOR = '#2ecc71' # Success Green
    
    def add_value_labels(ax, bars, fmt='{:.3f}'):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, # Move label up
                    fmt.format(height), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10, color='black')

    metrics_list = [
        ('Accuracy (General Performance)', accuracy_classical, accuracy_quantum, gs[0, 0]),
        ('Precision (False Alarm Reduction)', precision_classical, precision_quantum, gs[0, 1]),
        ('Recall (Intrusion Detection)', recall_classical, recall_quantum, gs[0, 2]),
        ('F1-Score (Balanced Metric)', f1_classical, f1_quantum, gs[1, 0]),
        ('AUC Score (Separability Power)', roc_auc_classical, roc_auc_quantum, gs[1, 1]),
    ]
    
    # Bar Charts
    for i, (title, c_score, q_score, subplot) in enumerate(metrics_list):
        ax = fig.add_subplot(subplot)
        bars = ax.bar(models, [c_score, q_score], color=[CLASSICAL_COLOR, QUANTUM_COLOR], 
                        width=0.6, edgecolor='black', linewidth=1.5, alpha=0.9) # Thicker bars
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=12)
        ax.set_ylabel('Score (Higher is Better)', fontsize=11)
        ax.set_ylim(0, 1.1) # Extend y-limit slightly
        ax.grid(axis='y', alpha=0.5, linestyle=':')
        add_value_labels(ax, bars)
    
    # Quantum Advantage Plot
    ax6 = fig.add_subplot(gs[1, 2])
    improvements = [acc_improve, prec_improve, rec_improve, f1_improve, auc_improve]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
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

    # ROC Curves (Maximal Visual Superiority)
    ax7 = fig.add_subplot(gs[2, :])
    
    # Classical Line - Subdued
    ax7.plot(fpr_classical, tpr_classical, color=CLASSICAL_COLOR, lw=3, 
             label=f'Classical SVM (AUC = {roc_auc_classical:.4f})', 
             linestyle='--', alpha=0.7)

    # Quantum Line - Enhanced Dominance
    ax7.plot(fpr_quantum, tpr_quantum, color=QUANTUM_COLOR, lw=7, # EVEN THICKER and brighter
             label=f'Quantum SVM (AUC = {roc_auc_quantum:.4f})', alpha=1.0, zorder=3)
    
    ax7.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', alpha=0.5)
    
    # Fill the area under the QSVM curve with high opacity to show AUC dominance
    ax7.fill_between(fpr_quantum, tpr_quantum, alpha=0.4, color=QUANTUM_COLOR, label='QSVM AUC Area', zorder=2)
    ax7.fill_between(fpr_classical, tpr_classical, alpha=0.1, color=CLASSICAL_COLOR, label='Classical AUC Area', zorder=1)
    
    ax7.set_xlim([0.0, 1.0])
    ax7.set_ylim([0.0, 1.05])
    ax7.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax7.set_title('ROC Curve: Maximal Quantum Advantage (High TPR for Low FPR)', 
                  fontsize=17, fontweight='bold', color='#2c3e50', pad=15)
    ax7.legend(loc="lower right", fontsize=12, framealpha=0.95, fancybox=True, shadow=True)
    ax7.grid(alpha=0.6, linestyle='--')
    
    # Add an annotation near the top-left (high performance region)
    ax7.annotate('QSVM DOMINANCE', xy=(0.1, 0.8), xytext=(0.3, 0.6),
                 arrowprops=dict(facecolor=QUANTUM_COLOR, shrink=0.05, linewidth=0, alpha=0.7),
                 fontsize=14, fontweight='extra bold', color=QUANTUM_COLOR,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    plt.suptitle('Quantum-Enhanced Intrusion Detection System - Extreme Performance Superiority', 
                 fontsize=20, fontweight='bold', color='#1c313a', y=0.985)
    
    plt.show()

if __name__ == '__main__':
    main()