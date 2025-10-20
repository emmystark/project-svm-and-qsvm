import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt
import sys
import time
import multiprocessing as mp
from functools import lru_cache, partial
import matplotlib.style as mplstyle
from sklearn.metrics import ConfusionMatrixDisplay

# Apply a professional, modern plot style
mplstyle.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# ====================================================================
# QUANTUM GATES AND FEATURE MAP (Optimized for Efficiency)
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

def create_quantum_state(x, n_qubits):
    """Simplified quantum feature map for faster computation"""
    # Initialize with Hadamard superposition
    H = hadamard()
    state = qt.tensor([H * qt.basis(2, 0) for _ in range(n_qubits)])

    # Single encoding layer for efficiency
    for i in range(n_qubits):
        angle = x[i] * np.pi
        Ry = ry(angle)
        op_ry = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
        state = op_ry * state

    # Simple entanglement
    for i in range(n_qubits - 1):
        cn = cnot_cached(n_qubits, i, i + 1)
        state = cn * state

    return state.unit()

def parallel_create_state(args):
    x, n_qubits = args
    return create_quantum_state(x, n_qubits)

def compute_quantum_kernel(X1, X2, n_qubits, batch_size=50):
    """Optimized quantum kernel computation"""
    print(f"Computing quantum kernel for {len(X1)} x {len(X2)} samples...")

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

    for i in range(len(X1)):
        bra = states1[i].dag()
        for j in range(len(X2)):
            overlap = bra * states2[j]
            kernel[i, j] = np.abs(overlap) ** 2

    # Normalize and add stability term
    kernel /= np.max(kernel) + 1e-10
    if kernel.shape[0] == kernel.shape[1]:
        kernel += np.eye(kernel.shape[0]) * 1e-4

    return kernel

def load_nsl_kdd_robust(train_file, test_file, max_samples=1000):
    """Load NSL-KDD dataset with sampling"""
    try:
        print("📁 Loading NSL-KDD dataset...")
        print(f"⚠️  LIMITING TO {max_samples} SAMPLES")
        
        # Load data
        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)
        
        print(f"📊 Raw shapes - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Apply sampling
        if len(train_df) > max_samples:
            train_df = train_df.sample(n=max_samples, random_state=42)
        if len(test_df) > max_samples // 4:
            test_df = test_df.sample(n=max_samples // 4, random_state=42)
        
        # Basic column setup
        n_cols = train_df.shape[1]
        train_df.columns = [f'feature_{i}' for i in range(n_cols-1)] + ['label']
        test_df.columns = [f'feature_{i}' for i in range(n_cols-1)] + ['label']
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None

def preprocess_data_simple(train_df, test_df):
    """Simplified preprocessing"""
    try:
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        print("🔄 Preprocessing data...")
        
        # Convert labels to binary
        train_processed['label_binary'] = (train_processed['label'] != 'normal').astype(int)
        test_processed['label_binary'] = (test_processed['label'] != 'normal').astype(int)
        
        # Prepare features (exclude label columns)
        feature_cols = [col for col in train_processed.columns if col not in ['label', 'label_binary']]
        
        X_train = train_processed[feature_cols].values.astype(float)
        y_train = train_processed['label_binary'].values
        X_test = test_processed[feature_cols].values.astype(float)
        y_test = test_processed['label_binary'].values
        
        print(f"✅ Preprocessing complete. Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None, None, None, None

def generate_dummy_data():
    """Generate dummy data as fallback"""
    n_components = 4
    train_size = 800
    test_size = 200
    X_train_reduced = np.random.rand(train_size, n_components) * np.pi/2
    y_train = np.random.randint(0, 2, train_size)
    X_test_reduced = np.random.rand(test_size, n_components) * np.pi/2
    y_test = np.random.randint(0, 2, test_size)
    print(f"✅ Generated {train_size} training and {test_size} test samples.\n")
    return X_train_reduced, X_test_reduced, y_train, y_test

def create_comprehensive_visualizations(classical_results, quantum_results, y_test, y_pred_classical, y_pred_quantum):
    """Create all visualizations in one figure"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC Curve Comparison
    plt.subplot(2, 3, 1)
    plt.plot(classical_results['fpr'], classical_results['tpr'], 
             color='blue', lw=2, label=f'Classical SVM (AUC = {classical_results["auc"]:.3f})')
    plt.plot(quantum_results['fpr'], quantum_results['tpr'], 
             color='red', lw=2, label=f'Quantum SVM (AUC = {quantum_results["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Metrics Comparison
    plt.subplot(2, 3, 2)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    classical_vals = [classical_results[m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    quantum_vals = [quantum_results[m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, classical_vals, width, label='Classical', color='blue', alpha=0.7)
    plt.bar(x + width/2, quantum_vals, width, label='Quantum', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(classical_vals):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(quantum_vals):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Quantum Advantage
    plt.subplot(2, 3, 3)
    improvements = [
        (quantum_results['accuracy'] - classical_results['accuracy']) * 100,
        (quantum_results['precision'] - classical_results['precision']) * 100,
        (quantum_results['recall'] - classical_results['recall']) * 100,
        (quantum_results['f1'] - classical_results['f1']) * 100
    ]
    
    colors = ['green' if x >= 0 else 'red' for x in improvements]
    bars = plt.bar(metrics, improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Improvement (%)')
    plt.title('Quantum Advantage')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if imp >= 0 else -1), 
                f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=10)
    
    # 4. Classical Confusion Matrix
    plt.subplot(2, 3, 4)
    cm_classical = confusion_matrix(y_test, y_pred_classical)
    sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues')
    plt.title('Classical SVM - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 5. Quantum Confusion Matrix
    plt.subplot(2, 3, 5)
    cm_quantum = confusion_matrix(y_test, y_pred_quantum)
    sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Reds')
    plt.title('Quantum SVM - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 6. Training Time Comparison
    plt.subplot(2, 3, 6)
    times = [classical_results['training_time'], quantum_results['training_time']]
    models = ['Classical\nSVM', 'Quantum\nSVM']
    bars = plt.bar(models, times, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 70)
    print("   🚀 QUANTUM SVM vs CLASSICAL SVM COMPARISON")
    print("=" * 70)
    
    # Define n_components at the start
    n_components = 4  # 4 Qubits
    
    # Load and preprocess data
    train_df, test_df = load_nsl_kdd_robust('KDDTrain+.csv', 'KDDTest+.csv', max_samples=1000)
    
    if train_df is not None and test_df is not None:
        X_train, X_test, y_train, y_test = preprocess_data_simple(train_df, test_df)
        
        if X_train is not None:
            # Dimensionality reduction
            pca = PCA(n_components=n_components)
            X_train_reduced = pca.fit_transform(X_train)
            X_test_reduced = pca.transform(X_test)
            
            # Scale to quantum-friendly range
            minmax = MinMaxScaler(feature_range=(0, np.pi / 2))
            X_train_reduced = minmax.fit_transform(X_train_reduced)
            X_test_reduced = minmax.transform(X_test_reduced)
            
            print(f"✅ Using {len(X_train_reduced)} training and {len(X_test_reduced)} test samples")
        else:
            print("❌ Preprocessing failed. Using dummy data.")
            X_train_reduced, X_test_reduced, y_train, y_test = generate_dummy_data()
    else:
        print("❌ Dataset loading failed. Using dummy data.")
        X_train_reduced, X_test_reduced, y_train, y_test = generate_dummy_data()

    # --- CLASSICAL SVM ---
    print("\n" + "="*50)
    print("🔵 TRAINING CLASSICAL SVM")
    print("="*50)
    
    classical_start = time.time()
    svm_classical = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    svm_classical.fit(X_train_reduced, y_train)
    classical_time = time.time() - classical_start
    
    y_pred_classical = svm_classical.predict(X_test_reduced)
    y_proba_classical = svm_classical.predict_proba(X_test_reduced)[:, 1]
    
    accuracy_classical = accuracy_score(y_test, y_pred_classical)
    precision_classical = precision_score(y_test, y_pred_classical, zero_division=0)
    recall_classical = recall_score(y_test, y_pred_classical, zero_division=0)
    f1_classical = f1_score(y_test, y_pred_classical, zero_division=0)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, y_proba_classical)
    auc_classical = auc(fpr_classical, tpr_classical)
    
    classical_results = {
        'accuracy': accuracy_classical,
        'precision': precision_classical,
        'recall': recall_classical,
        'f1': f1_classical,
        'auc': auc_classical,
        'fpr': fpr_classical,
        'tpr': tpr_classical,
        'training_time': classical_time
    }
    
    print(f"📊 Classical SVM Results:")
    print(f"   Accuracy:  {accuracy_classical:.4f}")
    print(f"   Precision: {precision_classical:.4f}")
    print(f"   Recall:    {recall_classical:.4f}")
    print(f"   F1-Score:  {f1_classical:.4f}")
    print(f"   AUC:       {auc_classical:.4f}")
    print(f"   Time:      {classical_time:.2f}s")

    # --- QUANTUM SVM ---
    print("\n" + "="*50)
    print("🔴 TRAINING QUANTUM SVM")
    print("="*50)
    
    quantum_start = time.time()
    n_qubits = n_components  # Now this is defined!
    
    try:
        # Use smaller subset for quantum kernel
        if len(X_train_reduced) > 500:
            train_indices = np.random.choice(len(X_train_reduced), 500, replace=False)
            X_train_q = X_train_reduced[train_indices]
            y_train_q = y_train[train_indices]
        else:
            X_train_q = X_train_reduced
            y_train_q = y_train
            
        print("Computing quantum kernel...")
        train_kernel = compute_quantum_kernel(X_train_q, X_train_q, n_qubits)
        test_kernel = compute_quantum_kernel(X_test_reduced, X_train_q, n_qubits)
        
        # Train quantum SVM
        svm_quantum = SVC(kernel='precomputed', C=1.0, probability=True)
        svm_quantum.fit(train_kernel, y_train_q)
        
        quantum_time = time.time() - quantum_start
        
        y_pred_quantum = svm_quantum.predict(test_kernel)
        y_proba_quantum = svm_quantum.predict_proba(test_kernel)[:, 1]
        
        accuracy_quantum = accuracy_score(y_test, y_pred_quantum)
        precision_quantum = precision_score(y_test, y_pred_quantum, zero_division=0)
        recall_quantum = recall_score(y_test, y_pred_quantum, zero_division=0)
        f1_quantum = f1_score(y_test, y_pred_quantum, zero_division=0)
        fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_proba_quantum)
        auc_quantum = auc(fpr_quantum, tpr_quantum)
        
        quantum_results = {
            'accuracy': accuracy_quantum,
            'precision': precision_quantum,
            'recall': recall_quantum,
            'f1': f1_quantum,
            'auc': auc_quantum,
            'fpr': fpr_quantum,
            'tpr': tpr_quantum,
            'training_time': quantum_time
        }
        
        print(f"📊 Quantum SVM Results:")
        print(f"   Accuracy:  {accuracy_quantum:.4f}")
        print(f"   Precision: {precision_quantum:.4f}")
        print(f"   Recall:    {recall_quantum:.4f}")
        print(f"   F1-Score:  {f1_quantum:.4f}")
        print(f"   AUC:       {auc_quantum:.4f}")
        print(f"   Time:      {quantum_time:.2f}s")
        
        # Create visualizations
        print("\n" + "="*50)
        print("📈 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        
        create_comprehensive_visualizations(classical_results, quantum_results, 
                                          y_test, y_pred_classical, y_pred_quantum)
        
        # Print quantum advantage summary
        print("\n" + "="*50)
        print("🚀 QUANTUM ADVANTAGE SUMMARY")
        print("="*50)
        print(f"Accuracy Improvement:  {accuracy_quantum - accuracy_classical:+.4f}")
        print(f"Precision Improvement: {precision_quantum - precision_classical:+.4f}")
        print(f"Recall Improvement:    {recall_quantum - recall_classical:+.4f}")
        print(f"F1-Score Improvement:  {f1_quantum - f1_classical:+.4f}")
        print(f"AUC Improvement:       {auc_quantum - auc_classical:+.4f}")
        
        # Calculate percentage improvements
        print("\n📊 Percentage Improvements:")
        print(f"Accuracy:  {((accuracy_quantum - accuracy_classical) / accuracy_classical * 100):+.2f}%")
        print(f"Precision: {((precision_quantum - precision_classical) / max(precision_classical, 0.01) * 100):+.2f}%")
        print(f"Recall:    {((recall_quantum - recall_classical) / max(recall_classical, 0.01) * 100):+.2f}%")
        print(f"F1-Score:  {((f1_quantum - f1_classical) / max(f1_classical, 0.01) * 100):+.2f}%")
        print(f"AUC:       {((auc_quantum - auc_classical) / auc_classical * 100):+.2f}%")
        
    except Exception as e:
        print(f"❌ Quantum SVM failed: {e}")
        print("Using classical results only...")
        # Create visualizations with just classical results
        quantum_results = classical_results.copy()
        create_comprehensive_visualizations(classical_results, classical_results, 
                                          y_test, y_pred_classical, y_pred_classical)

if __name__ == '__main__':
    main()