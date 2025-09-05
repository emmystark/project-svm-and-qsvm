import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Define NSL-KDD columns (41 features + label)
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

# Load NSL-KDD dataset with error handling
try:
    train_data = pd.read_csv('KDDTrain+.txt', names=columns, delimiter=',')
    test_data = pd.read_csv('KDDTest+.txt', names=columns, delimiter=',')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory as this script.")
    print("Download them from https://www.unb.ca/cic/datasets/nsl.html")
    exit(1)

# Debug: Inspect dataset
print("First 5 rows of train_data:")
print(train_data.head())
print("\nData types before preprocessing:")
print(train_data.dtypes)

# Debug: Check label distribution before binarization
print("\nUnique labels before binarization:", train_data['label'].unique())
print("Label counts before binarization:", train_data['label'].value_counts())

# Convert numeric columns to correct types
numeric_cols = [col for col in columns if col not in ['protocol_type', 'service', 'flag', 'label']]
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

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    all_values = pd.concat([train_data[col], test_data[col]]).unique()
    le = LabelEncoder()
    le.fit(all_values)
    train_data[col] = le.transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Convert labels to binary (normal vs. attack)
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Debug: Check label distribution after binarization
print("\nUnique labels in y_train after binarization:", train_data['label'].unique())
print("Label counts in y_train after binarization:", train_data['label'].value_counts())

# Split features and labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality for quantum circuit
pca = PCA(n_components=4)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Subsample with stratification to maintain class balance
X_train_reduced, _, y_train, _ = train_test_split(X_train_reduced, y_train, train_size=1000, random_state=42, stratify=y_train)
X_test_reduced, _, y_test, _ = train_test_split(X_test_reduced, y_test, train_size=200, random_state=42, stratify=y_test)

# Debug: Check class distribution after subsampling
print("\nUnique labels in y_train after subsampling:", np.unique(y_train))
print("Label counts in y_train after subsampling:", pd.Series(y_train).value_counts())

# Check for single-class issue
if len(np.unique(y_train)) < 2:
    print("Error: y_train contains only one class. Try increasing train_size or using SMOTE.")
    exit(1)

# Optional: Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_reduced, y_train = smote.fit_resample(X_train_reduced, y_train)
print("\nLabel counts after SMOTE:", pd.Series(y_train).value_counts())

# Create quantum feature map
feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')

# Create quantum kernel
backend = AerSimulator()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Train QSVM
svm = SVC(kernel=quantum_kernel.evaluate)
svm.fit(X_train_reduced, y_train)

# Evaluate
y_pred = svm.predict(X_test_reduced)
accuracy = svm.score(X_test_reduced, y_test)
print(f"\nAccuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))