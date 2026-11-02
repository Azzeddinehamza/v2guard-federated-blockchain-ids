import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import io
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import flwr as fl
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

# Selected features 
selected_features = ['src_bytes', 'dst_bytes', 'dst_host_count', 'dst_host_srv_count', 'duration', 'service_Encoded',
                    'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate',
                    'count', 'proto_Encoded', 'dst_host_rerror_rate', 'dst_host_serror_rate',
                    'dst_host_srv_diff_host_rate', 'logged_in', 'wrong_fragment', 'hot', 'srv_count',
                    'num_file_creations', 'diff_srv_rate', 'dst_host_srv_serror_rate', 'flag_Encoded']

train_path = os.environ.get("NSL_KDD_PATH", "data/KDDCombined.txt")

# Blockchain
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
assert w3.is_connected(), "Failed to connect to Ganache"
with open('fl-blockchain/build/contracts/ModelAggregation.json', 'r') as f:
    contract_json = json.load(f)
contract_abi = contract_json['abi']
contract_address = os.environ["CONTRACT_ADDRESS"]  # set in .env
contract = w3.eth.contract(address=contract_address, abi=contract_abi)
account = w3.eth.accounts[0]
private_key = os.environ["PRIVATE_KEY"]  # NEVER commit a real key
weights_storage = {}

# transaction details
def print_transaction_details(tx_hash, description, receipt):
    print(f"\n=== Transaction Details: {description} ===")
    print(f"Transaction Hash: {tx_hash.hex()}")
    print(f"From: {receipt['from']}")
    print(f"To: {receipt['to']}")
    print(f"Status: {'Success' if receipt['status'] == 1 else 'Failed'}")
    print(f"Gas Used: {receipt['gasUsed']}")
    print(f"Block Number: {receipt['blockNumber']}")
    print(f"Transaction Index: {receipt['transactionIndex']}")

# Hash model weights
def hash_weights(weights):
    weights_bytes = pickle.dumps(weights)
    return hashlib.sha256(weights_bytes).hexdigest()


def load_and_preprocess_data():
    feature_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
                    "num_failed_logins","logged_in","num_compromised","root_shell","suAttempted","num_root","num_file_creations",
                    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
                    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                    "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    all_data = pd.read_csv(train_path, names=feature_names)
    all_data.drop(['difficulty'], axis=1, inplace=True)

   
    def change_label(df):
        df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'], 'Dos', inplace=True)
        df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'], 'R2L', inplace=True)
        df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'], 'Probe', inplace=True)
        df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'], 'U2R', inplace=True)
        df.label.replace(['normal'], 'normal', inplace=True)

    change_label(all_data)
    all_data['multiclass_label'] = all_data['label']
    all_data['binary_label'] = all_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encode categorical features
    le_service = LabelEncoder()
    all_data['service_Encoded'] = le_service.fit_transform(all_data['service'])
    le_proto = LabelEncoder()
    all_data['proto_Encoded'] = le_proto.fit_transform(all_data['protocol_type'])
    le_flag = LabelEncoder()
    all_data['flag_Encoded'] = le_flag.fit_transform(all_data['flag'])

    # Split data
    train_data, test_data = train_test_split(all_data, test_size=0.3, random_state=42, stratify=all_data['multiclass_label'])
    
    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train = train_data[selected_features]
    y_train_multi = train_data['multiclass_label']
    X_train_resampled, y_train_multi_resampled = ros.fit_resample(X_train, y_train_multi)
    train_data_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=selected_features),
                                    pd.DataFrame(y_train_multi_resampled, columns=['multiclass_label'])], axis=1)
    train_data_resampled['binary_label'] = train_data_resampled['multiclass_label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Scale
    numeric_cols = train_data_resampled[selected_features].select_dtypes(include='number').columns
    scaler = StandardScaler()
    train_data_resampled[numeric_cols] = scaler.fit_transform(train_data_resampled[numeric_cols])
    test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

    # Encode multiclass labels
    le = LabelEncoder()
    train_data_resampled['multiclass_encoded'] = le.fit_transform(train_data_resampled['multiclass_label'])
    test_data['multiclass_encoded'] = le.transform(test_data['multiclass_label'])

    X_train_multi = train_data_resampled[selected_features]
    y_train_multi = train_data_resampled['multiclass_encoded']
    X_test_multi = test_data[selected_features]
    y_test_multi = test_data['multiclass_encoded']

    # Compute class weights
    class_counts = Counter(y_train_multi)
    total = sum(class_counts.values())
    class_weights = {cls: total/(len(class_counts)*count) for cls, count in class_counts.items()}

    return X_train_multi, y_train_multi, X_test_multi, y_test_multi, le, class_weights

# Load data
X_train_multi, y_train_multi, X_test_multi, y_test_multi, label_encoder, class_weights = load_and_preprocess_data()

#  Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights)

# Flower client for federated learning
class NSLKDDClient(fl.client.NumPyClient):
    def __init__(self, model, X_train_m, y_train_m, X_test_m, y_test_m, class_weights):
        self.model = model
        self.X_train_m = X_train_m
        self.y_train_m = y_train_m
        self.X_test_m = X_test_m
        self.y_test_m = y_test_m
        self.class_weights = class_weights

    def get_parameters(self):
        buffer = io.BytesIO()
        pickle.dump(self.model, buffer)
        return [buffer.getvalue()]

    def set_parameters(self, parameters):
        buffer = io.BytesIO(parameters[0])
        self.model = pickle.load(buffer)

    # Train model and submit hash to blockchain
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train_m, self.y_train_m)
        weights = self.get_parameters()
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
            print_transaction_details(tx_hash, f"Submitting model update (hash: {weights_hash}) for client", receipt)
        except Exception as e:
            print(f"Blockchain transaction failed: {str(e)}")
            raise
        weights_storage[weights_hash] = weights
        return weights, len(self.X_train_m), {}

    # Evaluate model 
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        multi_pred = self.model.predict(self.X_test_m)
        acc = accuracy_score(self.y_test_m, multi_pred)
        multi_precision, multi_recall, multi_f1, _ = precision_recall_fscore_support(
            self.y_test_m, multi_pred, average=None, zero_division=0
        )
        metrics = {
            "accuracy": float(acc),
            "precision": multi_precision.tolist(),
            "recall": multi_recall.tolist(),
            "f1": multi_f1.tolist(),
            "class_names": label_encoder.classes_.tolist()
        }
        return -float(acc), len(self.X_test_m), metrics

# Split data 
num_clients = 6
skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
clients_data = []
ros = RandomOverSampler(random_state=42)
X_train_multi_np = X_train_multi.to_numpy()
y_train_multi_np = y_train_multi.to_numpy()

# Ensure all classes in each shard
for train_index, _ in skf.split(X_train_multi_np, y_train_multi_np):
    X_shard = X_train_multi_np[train_index]
    y_shard = y_train_multi_np[train_index]
    X_shard_df = pd.DataFrame(X_shard, columns=selected_features)
    y_shard_series = pd.Series(y_shard)
    unique_classes = np.unique(y_shard)
    if len(unique_classes) < len(label_encoder.classes_):
        missing_classes = set(range(len(label_encoder.classes_))) - set(unique_classes)
        for missing_class in missing_classes:
            missing_class_samples = X_train_multi_np[y_train_multi_np == missing_class]
            missing_class_labels = y_train_multi_np[y_train_multi_np == missing_class]
            if len(missing_class_samples) > 0:
                num_samples_to_add = min(5, len(missing_class_samples))
                indices_to_add = np.random.choice(len(missing_class_samples), num_samples_to_add, replace=False)
                X_shard = np.vstack([X_shard, missing_class_samples[indices_to_add]])
                y_shard = np.hstack([y_shard, missing_class_labels[indices_to_add]])
    X_shard_df = pd.DataFrame(X_shard, columns=selected_features)
    y_shard_series = pd.Series(y_shard)
    X_shard_resampled, y_shard_resampled = ros.fit_resample(X_shard_df, y_shard_series)
    clients_data.append((X_shard_resampled, y_shard_resampled))

#  federated learning
def simulate_federated_learning():
    global rf_model, aggregated_pred
    num_rounds = 1
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        client_models = []
        client_hashes = []
        # Train each client
        for client_id in range(num_clients):
            print(f"Training client {client_id + 1}/{num_clients}")
            X_train_m, y_train_m = clients_data[client_id]
            client_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights
            )
            client = NSLKDDClient(client_model, X_train_m, y_train_m, X_test_multi, y_test_multi, class_weights)
            weights, _, _ = client.fit(client.get_parameters(), {})
            weights_hash = hash_weights(weights)
            client_models.append(weights)
            client_hashes.append(weights_hash)
        # verify blockchain updates
        try:
            updates = contract.functions.getUpdates(round_num).call()
            print(f"Retrieved {len(updates)} updates from blockchain for round {round_num}")
        except Exception as e:
            print(f"Failed to retrieve blockchain updates: {str(e)}")
            updates = []
        verified_models = []
        for update in updates:
            stored_hash = update[1]
            if stored_hash in weights_storage:
                computed_hash = hash_weights(weights_storage[stored_hash])
                if computed_hash == stored_hash:
                    verified_models.append(weights_storage[stored_hash])
                else:
                    print(f"Warning: Hash mismatch for update {stored_hash}")
        if not verified_models:
            print("No verified models for this round, using client models as fallback")
            verified_models = client_models
        # Aggregate predictions by majority voting
        predictions = []
        for model_weights in verified_models:
            buffer = io.BytesIO(model_weights[0])
            model = pickle.load(buffer)
            pred = model.predict(X_test_multi)
            predictions.append(pred)
        predictions = np.array(predictions)
        aggregated_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        # Advance blockchain round
        try:
            tx = contract.functions.nextRound().build_transaction({
                'from': account,
                'nonce': w3.eth.get_transaction_count(account),
                'gas': 2000000,
                'gasPrice': w3.to_wei('20', 'gwei')
            })
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            print_transaction_details(tx_hash, f"Advancing to next round (Round {round_num + 2})", receipt)
        except Exception as e:
            print(f"Failed to move to next round: {str(e)}")
            raise
        # Evaluate 
        acc = accuracy_score(y_test_multi, aggregated_predictions)
        multi_precision, multi_recall, multi_f1, _ = precision_recall_fscore_support(
            y_test_multi, aggregated_predictions, average=None, zero_division=0
        )
        metrics = {
            "accuracy": float(acc),
            "precision": multi_precision.tolist(),
            "recall": multi_recall.tolist(),
            "f1": multi_f1.tolist(),
            "class_names": label_encoder.classes_.tolist()
        }
        print("\nMulticlass Classification Metrics:")
        print(f"Global Accuracy: {metrics['accuracy']*100:.2f}%")
        for i, class_name in enumerate(metrics['class_names']):
            print(f"{class_name:8} - Precision: {metrics['precision'][i]:.3f}, Recall: {metrics['recall'][i]:.3f}, F1: {metrics['f1'][i]:.3f}")
        # Store aggregated predictions for final evaluation
        aggregated_pred = aggregated_predictions

# Print classification report
def print_detailed_report(y_true, y_pred, classes, title):
    print(f"\n{title}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Support': support
    })
    print("\nPer-class metrics:")
    print(metrics_df.to_string(index=False))

# Plot multiclass confusion matrix
def plot_multiclass_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Multiclass Confusion Matrix (%)\nGlobal Accuracy: {:.2f}%'.format(accuracy_score(y_true, y_pred)*100))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot binary confusion matrix
def plot_binary_confusion_matrix(y_true, y_pred, classes):
    binary_true = (y_true != label_encoder.transform(['normal'])[0]).astype(int)
    binary_pred = (y_pred != label_encoder.transform(['normal'])[0]).astype(int)
    cm = confusion_matrix(binary_true, binary_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Binary Confusion Matrix (%)\nAccuracy: {:.2f}%'.format(accuracy_score(binary_true, binary_pred)*100))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Print system resource usage
def print_system_resources():
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 3)
    used_memory = memory.used / (1024 ** 3)
    memory_percent = memory.percent
    cpu_percent = psutil.cpu_percent(interval=1)
    print("\n=== System Resource Usage ===")
    print(f"Total RAM: {total_memory:.2f} GB")
    print(f"Used RAM: {used_memory:.2f} GB ({memory_percent:.1f}%)")
    print(f"CPU Usage: {cpu_percent:.1f}%")

# Main execution
print("Starting Federated Learning for Multiclass Classification with Random Forest")
simulate_federated_learning()

# Final evaluation
print("\n=== Final Evaluation ===")
try:
    multi_pred = aggregated_pred
except NameError:
    print("Aggregated predictions not available, training on full dataset as fallback")
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
print_system_resources()
