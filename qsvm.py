
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import qutip as qt
from qutip_qip.operations import ry, rz, cnot
import sys
import time

def create_state(x, n_qubits):
    state = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    scale = 3.0  # Increased for better feature encoding
    for _ in range(2):  # Added a second layer for more expressiveness
        for i in range(n_qubits):
            Ry = ry(x[i] * scale)
            Rz = rz(x[i] * scale)
            op_ry = qt.tensor([Ry if j == i else qt.qeye(2) for j in range(n_qubits)])
            op_rz = qt.tensor([Rz if j == i else qt.qeye(2) for j in range(n_qubits)])
            state = op_ry * op_rz * state
        for i in range(n_qubits):
            cn = cnot(n_qubits, i, (i + 1) % n_qubits)
            state = cn * state
    return state

def compute_kernel(X1, X2, n_qubits):
    print(f"Computing kernel for {len(X1)} x {len(X2)} samples...")
    states1 = np.array([create_state(x, n_qubits).full().flatten() for x in X1], dtype=complex)
    states2 = np.array([create_state(x, n_qubits).full().flatten() for x in X2], dtype=complex)
    overlaps = np.einsum('id,jd->ij', states1.conj(), states2)
    return np.real(np.abs(overlaps) ** 2)

def main():
    print("Starting script execution at:", time.ctime())

    # Define NSL-KDD columns
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

    print("Loading datasets...")
    try:
        raw_train_data = pd.read_csv('KDDTrain+.txt', delimiter=',', header=None)
        raw_test_data = pd.read_csv('KDDTest+.txt', delimiter=',', header=None)
        print(f"Raw train data shape: {raw_train_data.shape}")
        print(f"Raw test data shape: {raw_test_data.shape}")

        if raw_train_data.shape[1] != 43:
            print(f"Error: Expected 43 columns, got {raw_train_data.shape[1]}. Check dataset format.")
            sys.exit(1)

        train_data = raw_train_data.copy()
        train_data.columns = columns
        test_data = raw_test_data.copy()
        test_data.columns = columns
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory.")
        print("Download from https://www.unb.ca/cic/datasets/nsl.html")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during file loading: {e}")
        sys.exit(1)

    print("Inspecting raw labels...")
    print("\nRaw label column (first 10 values):")
    print(raw_train_data.iloc[:, 41].head(10))
    print("\nUnique raw labels:", raw_train_data.iloc[:, 41].unique())
    print("Raw label counts:", raw_train_data.iloc[:, 41].value_counts())

    print("Inspecting dataset...")
    print("\nFirst 5 rows of train_data:")
    print(train_data.head())
    print("\nData types before preprocessing:")
    print(train_data.dtypes)

    print("Checking label distribution...")
    print("\nUnique labels before binarization:", train_data['label'].unique())
    print("Label counts before binarization:", train_data['label'].value_counts())

    print("Preprocessing data...")
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

    print("Encoding categorical features...")
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        all_values = pd.concat([train_data[col], test_data[col]]).unique()
        le = LabelEncoder()
        le.fit(all_values)
        train_data[col] = le.transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

    print("Data types after preprocessing:")
    print(train_data.dtypes)

    print("Binarizing labels...")
    train_data['label'] = (train_data['label'] != 'normal').astype(int)
    test_data['label'] = (test_data['label'] != 'normal').astype(int)

    print("\nUnique labels in y_train after binarization:", train_data['label'].unique())
    print("Label counts in y_train after binarization:", train_data['label'].value_counts())

    print("Splitting features and labels...")
    X_train = train_data.drop(['label', 'difficulty_level'], axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop(['label', 'difficulty_level'], axis=1).values
    y_test = test_data['label'].values

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Reducing dimensionality...")
    n_components = 5  # Increased for better performance
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)

    print("Applying MinMax scaling...")
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_train_reduced = minmax.fit_transform(X_train_reduced)
    X_test_reduced = minmax.transform(X_test_reduced)

    print("Subsampling data...")
    train_size = 200  # Balanced for speed and performance
    test_size = 100
    X_train_reduced, _, y_train, _ = train_test_split(X_train_reduced, y_train, train_size=train_size, random_state=42, stratify=y_train)
    X_test_reduced, _, y_test, _ = train_test_split(X_test_reduced, y_test, train_size=test_size, random_state=42, stratify=y_test)

    print("\nUnique labels in y_train after subsampling:", np.unique(y_train))
    print("Label counts in y_train after subsampling:", pd.Series(y_train).value_counts())

    if len(np.unique(y_train)) < 2:
        print("Error: y_train contains only one class. Try increasing train_size or check label binarization.")
        sys.exit(1)

    print("Training classical SVM...")
    svm_classical = SVC(kernel='rbf', random_state=42)
    param_grid_classical = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1]}
    grid_classical = GridSearchCV(svm_classical, param_grid_classical, cv=3)
    grid_classical.fit(X_train_reduced, y_train)
    svm_classical = grid_classical.best_estimator_
    print(f"Best Classical SVM parameters: {grid_classical.best_params_}")

    print("Evaluating classical SVM...")
    y_pred_classical = svm_classical.predict(X_test_reduced)
    accuracy_classical = svm_classical.score(X_test_reduced, y_test)
    print(f"\nClassical SVM Accuracy: {accuracy_classical:.2f}")
    print("Classical SVM Classification Report:")
    print(classification_report(y_test, y_pred_classical))

    print("Computing ROC and AUC for classical SVM...")
    y_score_classical = svm_classical.decision_function(X_test_reduced)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, y_score_classical)
    roc_auc_classical = auc(fpr_classical, tpr_classical)
    print(f"Classical SVM AUC: {roc_auc_classical:.2f}")

    print("Starting QSVM computation...")
    n_qubits = n_components
    try:
        print("Computing training kernel...")
        train_kernel = compute_kernel(X1=X_train_reduced, X2=X_train_reduced, n_qubits=n_qubits)
        print("Computing test kernel...")
        test_kernel = compute_kernel(X1=X_test_reduced, X2=X_train_reduced, n_qubits=n_qubits)
    except KeyboardInterrupt:
        print("QSVM computation interrupted by user.")
        sys.exit(0)

    print("Training QSVM...")
    param_grid = {'C': [100, 200, 300]}
    grid = GridSearchCV(SVC(kernel='precomputed'), param_grid, cv=3)
    grid.fit(train_kernel, y_train)
    svm_quantum = grid.best_estimator_
    print(f"Best QSVM C parameter: {grid.best_params_['C']}")

    print("Evaluating QSVM...")
    y_pred_quantum = svm_quantum.predict(test_kernel)
    accuracy_quantum = svm_quantum.score(test_kernel, y_test)
    print(f"\nQuantum SVM Accuracy: {accuracy_quantum:.2f}")
    print("Quantum SVM Classification Report:")
    print(classification_report(y_test, y_pred_quantum))

    print("Computing ROC and AUC for QSVM...")
    y_score_quantum = svm_quantum.decision_function(test_kernel)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_score_quantum)
    roc_auc_quantum = auc(fpr_quantum, tpr_quantum)
    print(f"Quantum SVM AUC: {roc_auc_quantum:.2f}")

    print("Plotting ROC curves...")
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

    print("Script execution completed at:", time.ctime())

if __name__ == '__main__':
    main()
