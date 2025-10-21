import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_curve, auc, 
                             precision_score, recall_score, f1_score, 
                             confusion_matrix, accuracy_score, roc_auc_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import lime
import lime.lime_tabular
from datetime import datetime, timedelta
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [14, 10]

# ==================================================================
# QUANTUM KERNEL SIMULATOR
# ==================================================================
class EfficientQuantumKernel:
    def __init__(self, n_qubits=4, batch_size=1000, gamma=1.0):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.gamma = gamma
        self.progress_times = []

    def compute_quantum_kernel_batch(self, X1, X2):
        """Enhanced quantum-inspired kernel with parameterized rotation"""
        print(f"🔮 Computing quantum kernel for {len(X1)} x {len(X2)} samples...")
        n1, n2 = len(X1), len(X2)
        kernel = np.zeros((n1, n2))
        self.progress_times = []
        
        # Handle gamma='scale'
        gamma_val = self.gamma
        if isinstance(self.gamma, str) and self.gamma == 'scale':
            n_features = X1.shape[1]
            var = np.var(X1)
            gamma_val = 1.0 / (n_features * var) if var > 0 else 1.0
        
        total_batches = (n1 // self.batch_size + 1) * (n2 // self.batch_size + 1)
        with tqdm(total=total_batches, desc="Quantum Kernel Progress") as pbar:
            for i in range(0, n1, self.batch_size):
                i_end = min(i + self.batch_size, n1)
                batch_X1 = X1[i:i_end]
                start_time = time.time()
                for j in range(0, n2, self.batch_size):
                    j_end = min(j + self.batch_size, n2)
                    batch_X2 = X2[j:j_end]
                    X1_expanded = batch_X1[:, np.newaxis, :]
                    X2_expanded = batch_X2[np.newaxis, :, :]
                    differences = X1_expanded - X2_expanded
                    squared_distances = np.sum(differences ** 2, axis=2)
                    kernel_batch = np.exp(-gamma_val * squared_distances) * \
                                   np.cos(np.pi * gamma_val * np.sqrt(squared_distances))
                    kernel[i:i_end, j:j_end] = kernel_batch
                    pbar.update(1)
                self.progress_times.append(time.time() - start_time)
        
        kernel /= np.max(kernel) + 1e-10
        if kernel.shape[0] == kernel.shape[1]:
            kernel += np.eye(kernel.shape[0]) * 1e-8
        return kernel

# ==================================================================
# LOAD AND PREPROCESS NSL-KDD DATASET
# ==================================================================
def load_nslkdd_dataset(train_filepath, test_filepath, test_size=0.2, max_samples=5000, n_features=10):
    """Load and preprocess NSL-KDD dataset with subsampling and feature selection"""
    print("📂 Loading NSL-KDD dataset...")
    
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
        'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'attack_type', 'difficulty'
    ]
    
    df_train = pd.read_csv(train_filepath, header=None, names=columns)
    df_test = pd.read_csv(test_filepath, header=None, names=columns)
    
    normal_samples = df_train[df_train['attack_type'] == 'normal']
    attack_samples = df_train[df_train['attack_type'] != 'normal']
    n_samples_per_class = min(len(normal_samples), len(attack_samples), max_samples // 2)
    df_train = pd.concat([
        normal_samples.sample(n=n_samples_per_class, random_state=42),
        attack_samples.sample(n=n_samples_per_class, random_state=42)
    ])
    
    print(f"✅ Training samples loaded: {len(df_train)}")
    print(f"✅ Test samples loaded: {len(df_test)}")
    
    df_train['is_attack'] = (df_train['attack_type'] != 'normal').astype(int)
    df_test['is_attack'] = (df_test['attack_type'] != 'normal').astype(int)
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        le_dict[col] = le
    
    feature_cols = [col for col in df.columns if col not in 
                    ['attack_type', 'is_attack', 'difficulty']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['is_attack'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['is_attack'].values
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-n_features:]
    feature_cols = [feature_cols[i] for i in top_indices]
    
    X_train = X_train[:, top_indices]
    X_test = X_test[:, top_indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Selected top {n_features} features: {feature_cols}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training - Normal: {(y_train==0).sum()}, Attacks: {(y_train==1).sum()}")
    print(f"   Test - Normal: {(y_test==0).sum()}, Attacks: {(y_test==1).sum()}")
    print(f"   Attack ratio (Train): {y_train.mean()*100:.1f}%")
    print(f"   Attack ratio (Test): {y_test.mean()*100:.1f}%")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, df

def load_nslkdd_simple(csv_filepath, test_size=0.2, max_samples=5000, n_features=10):
    """Alternative: Load single NSL-KDD CSV file with subsampling and feature selection"""
    print("📂 Loading NSL-KDD dataset from single CSV...")
    
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
        'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'attack_type', 'difficulty'
    ]
    
    df = pd.read_csv(csv_filepath, header=None, names=columns)
    
    normal_samples = df[df['attack_type'] == 'normal']
    attack_samples = df[df['attack_type'] != 'normal']
    n_samples_per_class = min(len(normal_samples), len(attack_samples), max_samples // 2)
    df = pd.concat([
        normal_samples.sample(n=n_samples_per_class, random_state=42),
        attack_samples.sample(n=n_samples_per_class, random_state=42)
    ])
    
    df['is_attack'] = (df['attack_type'] != 'normal').astype(int)
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    feature_cols = [col for col in df.columns if col not in 
                    ['attack_type', 'is_attack', 'difficulty']]
    X = df[feature_cols].values
    y = df['is_attack'].values
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-n_features:]
    feature_cols = [feature_cols[i] for i in top_indices]
    
    X = X[:, top_indices]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"✅ Selected top {n_features} features: {feature_cols}")
    print(f"✅ Dataset loaded: {len(X_train)} training, {len(X_test)} test samples")
    print(f"   Total samples: {len(df)}")
    print(f"   Attack ratio: {y.mean()*100:.1f}%")
    
    return X_train, X_test, y_train, y_test, feature_cols, df

# ==================================================================
# ATTACK ANALYSIS & STATISTICS
# ==================================================================
def analyze_attacks(df):
    """Comprehensive attack analysis"""
    print("\n" + "="*70)
    print("🔍 ATTACK ANALYSIS")
    print("="*70)
    
    attack_counts = df['attack_type'].value_counts()
    print("\n📊 Attack Type Distribution:")
    print("-" * 50)
    for attack, count in attack_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"  {attack:20} : {count:6} ({pct:5.1f}%)")
    
    print(f"\n📈 Total samples: {len(df)}")
    print(f"   Normal: {(df['is_attack']==0).sum()} ({(df['is_attack']==0).mean()*100:.1f}%)")
    print(f"   Attacks: {(df['is_attack']==1).sum()} ({(df['is_attack']==1).mean()*100:.1f}%)")

# ==================================================================
# LIME EXPLAINABILITY
# ==================================================================
def explain_predictions_with_lime(model, X_train, X_test, y_test, 
                                 feature_names, model_name, sample_idx=0):
    """Generate LIME explanations for model predictions"""
    print(f"\n🔬 Generating LIME explanations for {model_name}...")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=['Normal', 'Attack'],
        mode='classification', verbose=0
    )
    
    explanations = []
    for idx in [sample_idx, sample_idx + 1, sample_idx + 2]:
        if idx < len(X_test):
            exp = explainer.explain_instance(
                X_test[idx], model.predict_proba, num_features=10, num_samples=1000
            )
            explanations.append(exp)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'LIME Explanations - {model_name}', fontsize=14, fontweight='bold')
    
    for idx, exp in enumerate(explanations):
        ax = axes[idx]
        weights = exp.as_list()
        bars = ax.barh([w[0] for w in weights], [w[1] for w in weights], color='#2E86AB')
        ax.set_title(f'Sample {sample_idx + idx}\nPred: {"Attack" if model.predict([X_test[sample_idx + idx]])[0] else "Normal"} | Actual: {"Attack" if y_test[sample_idx + idx] else "Normal"}')
        ax.set_xlabel('Feature Contribution')
    
    plt.tight_layout()
    plt.savefig(f'lime_explanations_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ==================================================================
# PERFORMANCE METRICS & VISUALIZATION
# ==================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION METRICS - {model_name}")
    print(f"{'='*70}")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print(f"✅ ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'y_pred': y_pred, 'y_pred_proba': y_pred_proba
    }

# ==================================================================
# VISUALIZATION FUNCTIONS
# ==================================================================
def plot_roc_curves(metrics_svm, metrics_qsvm):
    """Plot ROC curves for both models"""
    fig = plt.figure(figsize=(10, 8))
    plt.plot(metrics_svm['fpr'], metrics_svm['tpr'], 
             label=f"SVM (AUC = {metrics_svm['roc_auc']:.4f})", 
             linewidth=2.5, color='#2E86AB')
    plt.plot(metrics_qsvm['fpr'], metrics_qsvm['tpr'], 
             label=f"QSVM (AUC = {metrics_qsvm['roc_auc']:.4f})", 
             linewidth=2.5, color='#A23B72')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve Comparison: SVM vs QSVM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_comparison(metrics_svm, metrics_qsvm):
    """Compare metrics between SVM and QSVM"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Metrics Comparison: SVM vs QSVM', fontsize=16, fontweight='bold')
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(2)
        values = [metrics_svm[metric], metrics_qsvm[metric]]
        bars = ax.bar(x, values, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.4)
        ax.set_ylabel(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['SVM', 'QSVM'])
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.4f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_confusion_matrices(metrics_svm, metrics_qsvm, y_test):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_svm = confusion_matrix(y_test, metrics_svm['y_pred'])
    cm_qsvm = confusion_matrix(y_test, metrics_qsvm['y_pred'])
    
    for idx, (cm, title) in enumerate([(cm_svm, 'SVM'), (cm_qsvm, 'QSVM')]):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        axes[idx].set_title(f'Confusion Matrix - {title}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_attack_statistics(df):
    """Visualize attack statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Attack Statistics & Distribution', fontsize=16, fontweight='bold')
    
    top_attacks = df['attack_type'].value_counts().head(10)
    axes[0, 0].barh(range(len(top_attacks)), top_attacks.values, color='#A23B72', alpha=0.8)
    axes[0, 0].set_yticks(range(len(top_attacks)))
    axes[0, 0].set_yticklabels(top_attacks.index)
    axes[0, 0].set_xlabel('Count', fontweight='bold')
    axes[0, 0].set_title('Top 10 Attack Types', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    labels = ['Normal', 'Attack']
    sizes = [(df['is_attack']==0).sum(), (df['is_attack']==1).sum()]
    colors = ['#2E86AB', '#A23B72']
    axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 1].set_title('Normal vs Attack Distribution', fontweight='bold')
    
    axes[1, 0].hist(df['duration'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Duration (seconds)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Connection Duration Distribution', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    service_attacks = df[df['is_attack']==1]['service'].value_counts().head(10)
    axes[1, 1].barh(range(len(service_attacks)), service_attacks.values, color='#E63946', alpha=0.8)
    axes[1, 1].set_yticks(range(len(service_attacks)))
    axes[1, 1].set_yticklabels(service_attacks.index)
    axes[1, 1].set_xlabel('Number of Attacks', fontweight='bold')
    axes[1, 1].set_title('Attacks by Service', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attack_statistics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_training_progress(svm_time, qsvm_time, qk_progress_times):
    """Visualize training progression for SVM and QSVM"""
    fig = plt.figure(figsize=(10, 6))
    
    plt.bar(['SVM'], [svm_time], color='#2E86AB', alpha=0.8, width=0.4, label='SVM Training Time')
    plt.bar(['QSVM'], [qsvm_time], color='#A23B72', alpha=0.8, width=0.4, label='QSVM Total Training Time')
    
    if qk_progress_times:
        batch_indices = np.arange(len(qk_progress_times)) + 1
        cumulative_times = np.cumsum(qk_progress_times)
        plt.plot(batch_indices, cumulative_times, marker='o', color='#E63946', 
                 label='QSVM Kernel Batch Progress')
    
    plt.xlabel('Model / Kernel Batch', fontsize=12, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Training Progression: SVM vs QSVM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==================================================================
# MAIN EXECUTION
# ==================================================================
def main():
    print("\n" + "="*70)
    print("🚀 ENHANCED INTRUSION DETECTION SYSTEM WITH LIME & ROC/AUC")
    print("   NSL-KDD Dataset (Limited to 5000 Training Samples)")
    print("="*70)
    
    print("\n📥 Select NSL-KDD loading method:")
    print("   1. Separate train/test files")
    print("   2. Single CSV file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        train_file = 'KDDTrain+.csv'
        test_file = 'KDDTest+.csv'
        
        print(f"\n📂 Loading NSL-KDD files...")
        print(f"   Train: {train_file}")
        print(f"   Test: {test_file}")
        
        if choice == '1':
            X_train, X_test, y_train, y_test, feature_names, df = load_nslkdd_dataset(
                train_file, test_file, max_samples=5000
            )
        else:
            X_train, X_test, y_train, y_test, feature_names, df = load_nslkdd_simple(
                train_file, max_samples=5000
            )
        
    except FileNotFoundError as e:
        print(f"⚠️ Error: {e}")
        print("   Please ensure the following files exist:")
        print("   - KDDTrain+.csv")
        print("   - KDDTest+.csv")
        print("   Download NSL-KDD from: https://www.unb.ca/cic/datasets/nsl-kdd/")
        return
    
    # Analyze attacks
    analyze_attacks(df)
    
    # Train SVM with progress tracking
    print(f"\n⏱️ Training SVM... ", end='')
    start = time.time()
    svm_model = SVC(kernel='rbf', probability=True, C=100, gamma='scale')
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - start
    print(f"✅ ({svm_time:.2f}s)")
    
    # Train QSVM with tuning
    print(f"⏱️ Training QSVM... ", end='')
    start = time.time()
    
    # Hyperparameter tuning for QSVM
    param_grid = {
        'C': [10, 100, 1000],
        'gamma': [0.01, 0.1, 1.0, 'scale']
    }
    best_qsvm_model = None
    best_score = 0
    best_params = None
    best_q_kernel_train = None
    
    for gamma in param_grid['gamma']:
        qk = EfficientQuantumKernel(n_qubits=4, batch_size=1000, gamma=gamma)
        q_kernel_train = qk.compute_quantum_kernel_batch(X_train, X_train)
        
        for C in param_grid['C']:
            qsvm_model = SVC(kernel='precomputed', probability=True, C=C)
            qsvm_model.fit(q_kernel_train, y_train)
            score = qsvm_model.score(q_kernel_train, y_train)
            if score > best_score:
                best_score = score
                best_qsvm_model = qsvm_model
                best_params = {'C': C, 'gamma': gamma}
                best_q_kernel_train = q_kernel_train
    
    print(f"\nBest QSVM params: C={best_params['C']}, gamma={best_params['gamma']}")
    qsvm_time = time.time() - start
    print(f"✅ ({qsvm_time:.2f}s)")
    
    # Compute test kernel for QSVM
    qk = EfficientQuantumKernel(n_qubits=4, batch_size=1000, gamma=best_params['gamma'])
    q_kernel_test = qk.compute_quantum_kernel_batch(X_test, X_train)
    
    # Evaluate models
    metrics_svm = evaluate_model(svm_model, X_test, y_test, "SVM")
    
    y_pred_qsvm = best_qsvm_model.predict(q_kernel_test)
    metrics_qsvm = {
        'accuracy': accuracy_score(y_test, y_pred_qsvm),
        'precision': precision_score(y_test, y_pred_qsvm, zero_division=0),
        'recall': recall_score(y_test, y_pred_qsvm, zero_division=0),
        'f1': f1_score(y_test, y_pred_qsvm, zero_division=0),
        'roc_auc': roc_auc_score(y_test, best_qsvm_model.decision_function(q_kernel_test)),
        'fpr': roc_curve(y_test, best_qsvm_model.decision_function(q_kernel_test))[0],
        'tpr': roc_curve(y_test, best_qsvm_model.decision_function(q_kernel_test))[1],
        'y_pred': y_pred_qsvm,
        'y_pred_proba': best_qsvm_model.decision_function(q_kernel_test)
    }
    
    print("\n" + "="*70)
    print("📊 QSVM METRICS")
    print("="*70)
    print(f"✅ Accuracy:  {metrics_qsvm['accuracy']:.4f}")
    print(f"✅ Precision: {metrics_qsvm['precision']:.4f}")
    print(f"✅ Recall:    {metrics_qsvm['recall']:.4f}")
    print(f"✅ F1-Score:  {metrics_qsvm['f1']:.4f}")
    print(f"✅ ROC-AUC:   {metrics_qsvm['roc_auc']:.4f}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_attack_statistics(df)
    plot_roc_curves(metrics_svm, metrics_qsvm)
    plot_metrics_comparison(metrics_svm, metrics_qsvm)
    plot_confusion_matrices(metrics_svm, metrics_qsvm, y_test)
    plot_training_progress(svm_time, qsvm_time, qk.progress_times)
    explain_predictions_with_lime(svm_model, X_train, X_test, y_test, feature_names, "SVM")
    
    print("\n✅ Analysis complete! Check generated PNG files.")

if __name__ == "__main__":
    main()