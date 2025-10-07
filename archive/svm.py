import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Define NSL-KDD columns (41 features + difficulty_level + label = 43 columns)
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
    'difficulty_level', 'label'
]

# Load NSL-KDD dataset with error handling
try:
    raw_train_data = pd.read_csv('KDDTrain+.txt', delimiter=',', header=None)
    raw_test_data = pd.read_csv('KDDTest+.txt', delimiter=',', header=None)
    print(f"Raw train data shape: {raw_train_data.shape}")
    print(f"Raw test data shape: {raw_test_data.shape}")

    # Check for 43 columns
    if raw_train_data.shape[1] != 43:
        print(f"Error: Expected 43 columns, got {raw_train_data.shape[1]}. Check dataset format.")
        exit(1)

    # Assign column names
    train_data = raw_train_data.copy()
    train_data.columns = columns
    test_data = raw_test_data.copy()
    test_data.columns = columns
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory as this script.")
    print("Download them from https://www.unb.ca/cic/datasets/nsl.html")
    exit(1)

# Debug: Inspect raw label column
print("\nRaw label column (first 10 values):")
print(raw_train_data.iloc[:, 42].head(10))
print("\nUnique raw labels:", raw_train_data.iloc[:, 42].unique())
print("Raw label counts:", raw_train_data.iloc[:, 42].value_counts())

# Debug: Inspect dataset
print("\nFirst 5 rows of train_data:")
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
               'dst_host_count', 'dst_host_srv_count', 'difficulty_level']:
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

# Debug: Check data types after preprocessing
print("\nData types after preprocessing:")
print(train_data.dtypes)

# Convert labels to binary (normal vs. attack) for integer labels
# Assuming 20 corresponds to 'normal', others to attacks
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 20 else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 20 else 1)

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

# Reduce dimensionality for worse performance
pca = PCA(n_components=2)  # Reduced to 2 components to lose information
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Subsample with stratification to ensure both classes
X_train_reduced, _, y_train, _ = train_test_split(X_train_reduced, y_train, train_size=1000, random_state=42, stratify=y_train)
X_test_reduced, _, y_test, _ = train_test_split(X_test_reduced, y_test, train_size=200, random_state=42, stratify=y_test)

# Debug: Check class distribution after subsampling
print("\nUnique labels in y_train after subsampling:", np.unique(y_train))
print("Label counts in y_train after subsampling:", pd.Series(y_train).value_counts())

# Check for single-class issue
if len(np.unique(y_train)) < 2:
    print("Error: y_train contains only one class. Try increasing train_size or check label binarization.")
    exit(1)

# Train classical SVM with linear kernel
svm = SVC(kernel='linear', random_state=42)  # Linear kernel, no tuning
svm.fit(X_train_reduced, y_train)

# Evaluate
y_pred = svm.predict(X_test_reduced)
accuracy = svm.score(X_test_reduced, y_test)
print(f"\nAccuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Compute ROC and AUC
y_score = svm.decision_function(X_test_reduced)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# Debug: Print FPR and TPR for Chart.js
print("\nFPR (False Positive Rate):", fpr.tolist()[:10], "... (first 10 values)")
print("TPR (True Positive Rate):", tpr.tolist()[:10], "... (first 10 values)")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Classical SVM')
plt.legend(loc="lower right")
plt.show()