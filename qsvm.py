import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
from qutip_qip.operations import ry, cnot

# Define NSL-KDD columns (41 features + label + difficulty_level = 43 columns)
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

# Load NSL-KDD dataset with error handling
try:
    raw_train_data = pd.read_csv('KDDTrain+.txt', delimiter=',', header=None)
    raw_test_data = pd.read_csv('KDDTest+.txt', delimiter=',', header=None)
    print(f"Raw train data shape: {raw_train_data.shape}")
    print(f"Raw test data shape: {raw_test_data.shape}")

    if raw_train_data.shape[1] != 43:
        print(f"Error: Expected 43 columns, got {raw_train_data.shape[1]}. Check dataset format.")
        exit(1)

    train_data = raw_train_data.copy()
    train_data.columns = columns
    test_data = raw_test_data.copy()
    test_data.columns = columns
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory.")
    print("Download from https://www.unb.ca/cic/datasets/nsl.html")
    exit(1)

# Debug: Inspect raw label column
print("\nRaw label column (first 10 values):")
print(raw_train_data.iloc[:, 41].head(10))
print("\nUnique raw labels:", raw_train_data.iloc[:, 41].unique())
print("Raw label counts:", raw_train_data.iloc[:, 41].value_counts())

# Debug: Inspect dataset
print("\nFirst 5 rows of train_data:")
print(train_data.head())
print("\nData types before preprocessing:")
print(train_data.dtypes)

# Debug: Check label distribution before binarization
print("\nUnique labels before binarization:", train_data['label'].unique())
print("Label counts before binarization:", train_data['label'].value_counts())

# Convert numeric columns to correct types
numeric_cols = [col for col in columns if col not in ['protocol_type', 'service', 'flag', 'label', 'difficulty_level']]
for col in numeric_cols:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0)
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0)
    if col in ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
               'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
               'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
               'dst_host_count', 'dst_host_srv_count']:
        train_data[col] = train_data[col].astype(int)
        test_data[col] = test_data[col].astype(int)
    else:
        train_data[col] = train_data[col].astype(float)
        test_data[col] = test_data[col].astype(float)

train_data['difficulty_level'] = pd.to_numeric(train_data['difficulty_level'], errors='coerce').fillna(0).astype(int)
test_data['difficulty_level'] = pd.to_numeric(test_data['difficulty_level'], errors='coerce').fillna(0).astype(int)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    all_values = pd.concat([train_data[col], test_data[col]]).unique()
    le = LabelEncoder()
    le.fit(all_values)
    train_data[col] = le.transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Debug: Check data types after preprocessing
print("\nData types after preprocessing:")
print(train_data.dtypes)

# Convert labels to binary (normal vs. attack)
train_data['label'] = (train_data['label'] != 'normal').astype(int)
test_data['label'] = (test_data['label'] != 'normal').astype(int)

# Debug: Check label distribution after binarization
print("\nUnique labels in y_train after binarization:", train_data['label'].unique())
print("Label counts in y_train after binarization:", train_data['label'].value_counts())

# Split features and labels
X_train = train_data.drop(['label', 'difficulty_level'], axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop(['label', 'difficulty_level'], axis=1).values
y_test = test_data['label'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality
n_components = 4
pca = PCA(n_components=n_components)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# MinMax scale for quantum angles
minmax = MinMaxScaler(feature_range=(0, np.pi))
X_train_reduced = minmax.fit_transform(X_train_reduced)
X_test_reduced = minmax.transform(X_test_reduced)

# Subsample with stratification
train_size = 1000
test_size = 200
X_train_reduced, _, y_train, _ = train_test_split(X_train_reduced, y_train, train_size=train_size, random_state=42, stratify=y_train)
X_test_reduced, _, y_test, _ = train_test_split(X_test_reduced, y_test, train_size=test_size, random_state=42, stratify=y_test)

# Debug: Check class distribution after subsampling
print("\nUnique labels in y_train after subsampling:", np.unique(y_train))
print("Label counts in y_train after subsampling:", pd.Series(y_train).value_counts())

# Check for single-class issue
if len(np.unique(y_train)) < 2:
    print("Error: y_train contains only one class. Try increasing train_size or check label binarization.")
    exit(1)

# Train classical SVM with linear kernel for comparison
svm_classical = SVC(kernel='linear', random_state=42)
svm_classical.fit(X_train_reduced, y_train)

# Evaluate classical
y_pred_classical = svm_classical.predict(X_test_reduced)
accuracy_classical = svm_classical.score(X_test_reduced, y_test)
print(f"\nClassical SVM Accuracy: {accuracy_classical:.2f}")
print("Classical SVM Classification Report:")
print(classification_report(y_test, y_pred_classical))

# Compute ROC and AUC for classical
y_score_classical = svm_classical.decision_function(X_test_reduced)
fpr_classical, tpr_classical, _ = roc_curve(y_test, y_score_classical)
roc_auc_classical = auc(fpr_classical, tpr_classical)
print(f"Classical SVM AUC: {roc_auc_classical:.2f}")

# QSVM part
def create_state(x, n_qubits):
    state = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    # Apply RY rotations
    for i in range(n_qubits):
        Ry = ry(x[i])  # Updated to use imported ry
        op = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
        state = op * state
    # Add entanglement with CNOT ring
    for i in range(n_qubits):
        cn = cnot(n_qubits, i, (i + 1) % n_qubits)  # Updated to use imported cnot
        state = cn * state
    return state

def compute_kernel(X1, X2, n_qubits):
    states1 = np.array([create_state(x, n_qubits).full().flatten() for x in X1], dtype=complex)
    states2 = np.array([create_state(x, n_qubits).full().flatten() for x in X2], dtype=complex)
    overlaps = np.einsum('id,jd->ij', states1.conj(), states2)
    return np.real(np.abs(overlaps) ** 2)

n_qubits = n_components
train_kernel = compute_kernel(X_train_reduced, X_train_reduced, n_qubits)
test_kernel = compute_kernel(X_test_reduced, X_train_reduced, n_qubits)

# Train QSVM
svm_quantum = SVC(kernel='precomputed', random_state=42, C=1.0)
svm_quantum.fit(train_kernel, y_train)

# Evaluate QSVM
y_pred_quantum = svm_quantum.predict(test_kernel)
accuracy_quantum = svm_quantum.score(test_kernel, y_test)
print(f"\nQuantum SVM Accuracy: {accuracy_quantum:.2f}")
print("Quantum SVM Classification Report:")
print(classification_report(y_test, y_pred_quantum))

# Compute ROC and AUC for QSVM
y_score_quantum = svm_quantum.decision_function(test_kernel)
fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_score_quantum)
roc_auc_quantum = auc(fpr_quantum, tpr_quantum)
print(f"Quantum SVM AUC: {roc_auc_quantum:.2f}")

# Plot ROC curves
plt.figure()
plt.plot(fpr_classical, tpr_classical, color='blue', lw=2, label=f'Classical ROC (AUC = {roc_auc_classical:.2f})')
plt.plot(fpr_quantum, tpr_quantum, color='darkorange', lw=2, label=f'Quantum ROC (AUC = {roc_auc_quantum:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()