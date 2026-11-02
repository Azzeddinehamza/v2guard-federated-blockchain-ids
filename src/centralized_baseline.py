import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import io
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from web3 import Web3
import hashlib
import json
import pickle
import os
import psutil
from sklearn.exceptions import NotFittedError
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# Selected features from kddRandomForest.py
selected_features = ['src_bytes', 'dst_bytes', 'dst_host_count', 'dst_host_srv_count', 'duration', 'service_Encoded',
                    'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate',
                    'count', 'proto_Encoded', 'dst_host_rerror_rate', 'dst_host_serror_rate',
                    'dst_host_srv_diff_host_rate', 'logged_in', 'wrong_fragment', 'hot', 'srv_count',
                    'num_file_creations', 'diff_srv_rate', 'dst_host_srv_serror_rate', 'flag_Encoded']

# Load only the training data
train_path = os.environ.get("NSL_KDD_PATH", "data/KDDCombined.txt")

# Blockchain setup
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
assert w3.is_connected(), "Failed to connect to Ganache"

# Load contract ABI
with open('fl-blockchain/build/contracts/ModelAggregation.json', 'r') as f:
    contract_json = json.load(f)
contract_abi = contract_json['abi']
contract_address = os.environ["CONTRACT_ADDRESS"]  # set in .env

# Initialize contract
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Use the first Ganache account and its private key
account = w3.eth.accounts[0]
private_key = os.environ["PRIVATE_KEY"]  # NEVER commit a real key
# Off-chain storage for model weights
weights_storage = {}

def hash_weights(weights):
    weights_bytes = pickle.dumps(weights)
    return hashlib.sha256(weights_bytes).hexdigest()

def get_parameters(model):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return [buffer.getvalue()]

def load_and_preprocess_data():
    feature_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
                    "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
                    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
                    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                    "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    all_data = pd.read_csv(train_path, names=feature_names)
    all_data.drop(['difficulty'], axis=1, inplace=True)

    def change_label(df):
        df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
        df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
        df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
        df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)
        df.label.replace(['normal'],'normal',inplace=True)

    change_label(all_data)
    all_data['multiclass_label'] = all_data['label']
    all_data['binary_label'] = all_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

    le_service = LabelEncoder()
    all_data['service_Encoded'] = le_service.fit_transform(all_data['service'])
    le_proto = LabelEncoder()
    all_data['proto_Encoded'] = le_proto.fit_transform(all_data['protocol_type'])
    le_flag = LabelEncoder()
    all_data['flag_Encoded'] = le_flag.fit_transform(all_data['flag'])

    train_data, test_data = train_test_split(all_data, test_size=0.3, random_state=42, stratify=all_data['multiclass_label'])
    ros = RandomOverSampler(random_state=42)
    X_train = train_data[selected_features]
    y_train_multi = train_data['multiclass_label']

    X_train_resampled, y_train_multi_resampled = ros.fit_resample(X_train, y_train_multi)
    train_data_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=selected_features),
                                    pd.DataFrame(y_train_multi_resampled, columns=['multiclass_label'])], axis=1)
    train_data_resampled['binary_label'] = train_data_resampled['multiclass_label'].apply(lambda x: 0 if x == 'normal' else 1)

    numeric_cols = train_data_resampled[selected_features].select_dtypes(include='number').columns
    scaler = StandardScaler()
    train_data_resampled[numeric_cols] = scaler.fit_transform(train_data_resampled[numeric_cols])
    test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

    le = LabelEncoder()
    train_data_resampled['multiclass_encoded'] = le.fit_transform(train_data_resampled['multiclass_label'])
    test_data['multiclass_encoded'] = le.transform(test_data['multiclass_label'])

    X_train_multi = train_data_resampled[selected_features]
    y_train_multi = train_data_resampled['multiclass_encoded']
    X_test_multi = test_data[selected_features]
    y_test_multi = test_data['multiclass_encoded']
   

    class_counts = Counter(y_train_multi)
    total = sum(class_counts.values())
    class_weights = {cls: total/(len(class_counts)*count) for cls, count in class_counts.items()}

    return (
            X_train_multi, y_train_multi, X_test_multi, y_test_multi, le, class_weights)

def print_detailed_report(y_true, y_pred, classes, title):
    print(f"\n{title}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Support': support
    })
    print("\nPer-class metrics:")
    print(metrics_df.to_string(index=False))

def plot_multiclass_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Multiclass Confusion Matrix (%)\nGlobal Accuracy: {:.2f}%'.format(accuracy_score(y_true, y_pred)*100))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_binary_confusion_matrix(y_true, y_pred, classes):
    binary_true = (y_true != label_encoder.transform(['normal'])[0]).astype(int)
    binary_pred = (y_pred != label_encoder.transform(['normal'])[0]).astype(int)
    cm = confusion_matrix(binary_true, binary_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Binary Confusion Matrix (%)\nAccuracy: {:.2f}%'.format(accuracy_score(binary_true, binary_pred)*100))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def print_system_resources():
    # Get RAM usage
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 3)  # Convert to GB
    used_memory = memory.used / (1024 ** 3)  # Convert to GB
    memory_percent = memory.percent

    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    print("\n=== System Resource Usage ===")
    print(f"Total RAM: {total_memory:.2f} GB")
    print(f"Used RAM: {used_memory:.2f} GB ({memory_percent:.1f}%)")
    print(f"CPU Usage: {cpu_percent:.1f}%")

# Main execution
print("Starting Random Forest Classification with Blockchain Integration for NSL-KDD Dataset")

# Load and preprocess data
(
 X_train_multi, y_train_multi, X_test_multi, y_test_multi, label_encoder, class_weights) = load_and_preprocess_data()

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)
print("Training Random Forest model...")
rf_model.fit(X_train_multi, y_train_multi)

# Store model weights in blockchain
weights = get_parameters(rf_model)
weights_hash = hash_weights(weights)
try:
    tx = contract.functions.submitUpdate(weights_hash).build_transaction({
        'from': account,
        'nonce': w3.eth.get_transaction_count(account),
        'gas': 2000000,
        'gasPrice': w3.to_wei('20', 'gwei')
    })
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Blockchain transaction successful: {tx_hash.hex()}")
except Exception as e:
    print(f"Blockchain transaction failed: {str(e)}")
    raise
weights_storage[weights_hash] = weights
print(f"Stored model weights with hash: {weights_hash}")

# Final evaluation
print("\n=== Final Evaluation ===")
try:
    multi_pred = rf_model.predict(X_test_multi)
except NotFittedError:
    print("Model not fitted, training on full dataset as fallback")
    rf_model.fit(X_train_multi, y_train_multi)
    multi_pred = rf_model.predict(X_test_multi)
class_names = label_encoder.classes_
print_detailed_report(y_test_multi, multi_pred, class_names, "Multiclass Classification Report")
binary_true = (y_test_multi != label_encoder.transform(['normal'])[0]).astype(int)
binary_pred = (multi_pred != label_encoder.transform(['normal'])[0]).astype(int)
print("\nBinary Classification Report (derived from multiclass):")
print_detailed_report(binary_true, binary_pred, ['Normal', 'Attack'], "Binary Classification Report")
plot_multiclass_confusion_matrix(y_test_multi, multi_pred, class_names)
plot_binary_confusion_matrix(y_test_multi, multi_pred, class_names)
final_accuracy = accuracy_score(y_test_multi, multi_pred)
print(f"\nFinal Global Multiclass Accuracy: {final_accuracy*100:.2f}%")
binary_accuracy = accuracy_score(binary_true, binary_pred)
print(f"Final Binary Accuracy (derived from multiclass): {binary_accuracy*100:.2f}%")

# Print system resource usage
print_system_resources()
