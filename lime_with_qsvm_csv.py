import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
import sys
import time
import multiprocessing as mp
from functools import lru_cache, partial
import matplotlib.style as mplstyle
from sklearn.metrics import ConfusionMatrixDisplay
from qsvm_lime_analysi import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

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

def load_nsl_kdd_robust(train_file, test_file):
    """
    Robust NSL-KDD dataset loading with dynamic column handling
    """
    try:
        print("📁 Loading NSL-KDD dataset from local files...")
        
        # Load data without assuming column structure
        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)
        
        print(f"📊 Raw data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Standard NSL-KDD column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        
        # Handle column mismatch dynamically
        n_train_cols = train_df.shape[1]
        n_test_cols = test_df.shape[1]
        
        if n_train_cols == 42 and n_test_cols == 42:
            # Standard case: 41 features + 1 label
            train_df.columns = columns
            test_df.columns = columns
        elif n_train_cols == 43 or n_test_cols == 43:
            # Handle extra column (usually difficulty score)
            extended_columns = columns + ['difficulty']
            if n_train_cols == 43:
                train_df.columns = extended_columns
            else:
                train_df.columns = columns[:41] + ['label']  # Truncate for train
                
            if n_test_cols == 43:
                test_df.columns = extended_columns
            else:
                test_df.columns = columns[:41] + ['label']  # Truncate for test
        else:
            # Fallback: use numeric column names
            print(f"⚠️  Non-standard column count. Train: {n_train_cols}, Test: {n_test_cols}")
            train_df.columns = [f'col_{i}' for i in range(n_train_cols)]
            test_df.columns = [f'col_{i}' for i in range(n_test_cols)]
            # Assume last column is label
            train_df.rename(columns={f'col_{n_train_cols-1}': 'label'}, inplace=True)
            test_df.rename(columns={f'col_{n_test_cols-1}': 'label'}, inplace=True)
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None

def preprocess_data(train_df, test_df):
    """
    Preprocess NSL-KDD data with robust error handling
    """
    try:
        # Make copies to avoid modifying originals
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        print("🔄 Preprocessing data...")
        
        # Ensure label column exists
        if 'label' not in train_processed.columns or 'label' not in test_processed.columns:
            print("❌ Label column not found in dataset")
            return None, None, None, None
        
        # Print data types for debugging
        print("Data types of each column in the dataset:")
        print(train_processed.dtypes)
        print("\n")
        
        # NEW: Analysis of Intrusion Types, Number of Attacks, and Time of Attack (Duration)
        print("=" * 70)
        print("🔍 DATASET ANALYSIS: INTRUSION TYPES, ATTACK COUNTS, AND DURATION STATS")
        print("=" * 70)

        # Types of intrusions (top 10 most frequent labels)
        print("Types of Intrusions (Top 10):")
        intrusion_types = train_processed['label'].value_counts().head(10)
        print(intrusion_types)
        print("\n")

        # Number of attacks
        num_normal = (train_processed['label'] == 'normal').sum()
        num_attacks = len(train_processed) - num_normal
        print(f"Total Records: {len(train_processed)}")
        print(f"Number of Normal Connections: {num_normal}")
        print(f"Number of Attacks: {num_attacks}")
        print(f"Attack Percentage: {(num_attacks / len(train_processed)) * 100:.2f}%")
        print("\n")

        # Time of attack (duration stats)
        if 'duration' in train_processed.columns:
            train_processed['duration'] = pd.to_numeric(train_processed['duration'], errors='coerce').fillna(0)
            test_processed['duration'] = pd.to_numeric(test_processed['duration'], errors='coerce').fillna(0)

            print("Duration Statistics (Time of Connections/Attacks):")
            print("Overall (Train):")
            print(train_processed['duration'].describe())
            print("\nBy Category (Normal vs Attack - Train):")
            duration_normal = train_processed[train_processed['label'] == 'normal']['duration'].describe()
            duration_attack = train_processed[train_processed['label'] != 'normal']['duration'].describe()
            print("Normal Connections:")
            print(duration_normal)
            print("\nAttack Connections:")
            print(duration_attack)

        # Identify numeric and categorical columns
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Ensure categorical columns exist
        categorical_cols = [col for col in categorical_cols if col in train_processed.columns]
        
        # Process numeric columns
        for col in numeric_cols:
            if col != 'label':  # Don't process label column
                train_processed[col] = pd.to_numeric(train_processed[col], errors='coerce').fillna(0)
                test_processed[col] = pd.to_numeric(test_processed[col], errors='coerce').fillna(0)
        
        # Process categorical columns with consistent encoding
        for col in categorical_cols:
            if col in train_processed.columns:
                # Combine train and test categories for consistent encoding
                combined_categories = pd.concat([
                    train_processed[col].astype(str), 
                    test_processed[col].astype(str)
                ]).unique()
                
                le = LabelEncoder()
                le.fit(combined_categories)
                
                train_processed[col] = le.transform(train_processed[col].astype(str))
                test_processed[col] = le.transform(test_processed[col].astype(str))
        
        # Convert labels to binary (normal vs attack)
        train_processed['label_binary'] = (train_processed['label'] != 'normal').astype(int)
        test_processed['label_binary'] = (test_processed['label'] != 'normal').astype(int)
        
        # Prepare feature matrices (exclude label columns)
        exclude_cols = ['label', 'label_binary', 'difficulty'] if 'difficulty' in train_processed.columns else ['label', 'label_binary']
        feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
        
        X_train = train_processed[feature_cols].values
        y_train = train_processed['label_binary'].values
        X_test = test_processed[feature_cols].values
        y_test = test_processed['label_binary'].values
        
        print(f"✅ Preprocessing complete. Features: {len(feature_cols)}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None, None, None, None

def generate_dummy_data():
    """Generate dummy data as fallback"""
    n_components = 4
    train_size = 300
    test_size = 100
    X_train_reduced = np.random.rand(train_size, n_components) * np.pi/2
    y_train = np.random.randint(0, 2, train_size)
    X_test_reduced = np.random.rand(test_size, n_components) * np.pi/2
    y_test = np.random.randint(0, 2, test_size)
    print(f"✅ Generated {train_size} training and {test_size} test samples.\n")
    return X_train_reduced, X_test_reduced, y_train, y_test

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    print("=" * 70)
    print("   🚀 OPTIMIZED QSVM - ENHANCED EFFICIENCY AND ACCURACY")
    print("=" * 70)
    qsvm_start = time.time()

    # Load and preprocess data
    train_df, test_df = load_nsl_kdd_robust('KDDTrain+.csv', 'KDDTest+.csv')
    
    if train_df is not None and test_df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
        
        if X_train is not None:
            # Apply feature scaling and dimensionality reduction
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

            print(f"✅ Using {len(X_train_reduced)} training and {len(X_test_reduced)} test samples from NSL-KDD dataset.\n")
        else:
            print("❌ Data preprocessing failed. Generating dummy data.")
            X_train_reduced, X_test_reduced, y_train, y_test = generate_dummy_data()
    else:
        print("❌ Dataset loading failed. Generating dummy data.")
        X_train_reduced, X_test_reduced, y_train, y_test = generate_dummy_data()

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
    # Add probability=True for LIME compatibility
    param_grid = {'C': [1, 10, 100, 500, 1000, 5000, 10000]}  # Expanded for better optimization
    grid = GridSearchCV(SVC(kernel='precomputed', probability=True), param_grid, cv=5, scoring='f1')  # Increased CV
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

    # --- CONFUSION MATRIX WITH GRAPH ---
    cm = confusion_matrix(y_test, y_pred_quantum)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Reds')
    plt.title('QSVM Confusion Matrix')
    plt.show()

    # ====================================================================
    # LIME INTERPRETABILITY FOR QSVM
    # ====================================================================
    print("=" * 70)
    print("🧠 LIME: LOCAL INTERPRETABILITY FOR QSVM")
    print("=" * 70)

    # Custom predictor wrapper for QSVM (computes kernel on-the-fly for perturbations)
    def qsvm_predict_proba(X):
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        K = compute_quantum_kernel_advanced(X, X_train_reduced, n_qubits)
        return svm_quantum.predict_proba(K)

    # Create LIME explainer
    feature_names = [f'Principal Component {i+1}' for i in range(X_train_reduced.shape[1])]
    explainer = LimeTabularExplainer(
        X_train_reduced,
        feature_names=feature_names,
        class_names=['Normal', 'Anomaly'],
        mode='classification',
        discretize_continuous=True
    )

    # Explain the first test instance (adjust index as needed)
    instance_idx = 0
    instance = X_test_reduced[instance_idx]
    prediction = y_pred_quantum[instance_idx]
    true_label = y_test[instance_idx]

    print(f"Explaining instance {instance_idx}: True Label = {true_label}, QSVM Prediction = {prediction}")

    lime_start = time.time()
    exp = explainer.explain_instance(
        instance,
        qsvm_predict_proba,
        num_features=4,  # All features since n=4
        num_samples=500  # Reduced for efficiency; increase for more accuracy
    )
    lime_time = time.time() - lime_start
    print(f"⏱️ LIME explanation time: {lime_time:.1f}s")

    # Visualize the explanation
    fig = exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for QSVM\nInstance {instance_idx}: True={true_label}, Pred={prediction}')
    plt.tight_layout()
    plt.show()

    # Print feature contributions
    print("\nFeature Contributions (LIME):")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")

if __name__ == '__main__':
    main()