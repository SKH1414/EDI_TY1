import json
import os

# Create config.py
with open("config.py", "w") as f:
    f.write("SEED = 42\n")

# Create split_data.py
split_data_code = """import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from config import SEED

def main():
    print("Loading dataset...")
    df = pd.read_parquet('cleaned_dataset.parquet')
    
    # Stratified split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.3, random_state=SEED, stratify=df['label']
    )
    
    # 50/50 split of the 30% temp gives 15% val, 15% test
    # Get labels for temp split to stratify again
    temp_labels = df.loc[temp_idx, 'label']
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=temp_labels
    )
    
    split_info = {
        'seed': SEED,
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist()
    }
    
    with open('split_indices.json', 'w') as f:
        json.dump(split_info, f)
        
    print(f"Saved splits to split_indices.json: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test.")

if __name__ == "__main__":
    main()
"""
with open("split_data.py", "w") as f:
    f.write(split_data_code)

try:
    import nbformat as nbf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat"])
    import nbformat as nbf

def write_notebook(filename, cells_data):
    nb = nbf.v4.new_notebook()
    for ctype, source in cells_data:
        if ctype == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        elif ctype == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
    with open(filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

# ----------------- RF Notebook ----------------- #
rf_cells = [
    ("markdown", "# Random Forest Pipeline\nImplements the Random Forest baseline as per `pipeline_design.txt` and prompt instructions."),
    ("code", "import pandas as pd\nimport numpy as np\nimport json\nimport os\nimport joblib\nimport time\nimport sklearn\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\nimport sys\n\nprint(f\"Python: {sys.version}\")\nprint(f\"pandas: {pd.__version__}\")\nprint(f\"sklearn: {sklearn.__version__}\")\nprint(f\"joblib: {joblib.__version__}\")"),
    ("code", "sys.path.append('.')\nfrom config import SEED\nprint(f\"Shared SEED: {SEED}\")"),
    ("code", "out_dir = 'outputs/rf'\nos.makedirs(out_dir, exist_ok=True)"),
    ("code", "# Load data and splits\nprint(\"Loading data...\")\ndf = pd.read_parquet('cleaned_dataset.parquet')\nfeature_cols = [c for c in df.columns if c != 'label']\n\nwith open('split_indices.json', 'r') as f:\n    splits = json.load(f)\n\nassert splits['seed'] == SEED, \"Seed mismatch between config.py and split_indices.json\"\n\nX_train = df.loc[splits['train_idx'], feature_cols].values\ny_train = df.loc[splits['train_idx'], 'label'].values\nX_val = df.loc[splits['val_idx'], feature_cols].values\ny_val = df.loc[splits['val_idx'], 'label'].values\nX_test = df.loc[splits['test_idx'], feature_cols].values\ny_test = df.loc[splits['test_idx'], 'label'].values\n\nprint(f\"Train shapes: X={X_train.shape}, y={y_train.shape}\")\nprint(f\"Val shapes: X={X_val.shape}, y={y_val.shape}\")\nprint(f\"Test shapes: X={X_test.shape}, y={y_test.shape}\")"),
    ("code", "# Preprocessing\n# No preprocessing needed for RF (no scaling)\n# Just save a null/no-op scaler to disk for uniformity\nscaler = None\njoblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))\nprint(\"Null scaler saved.\")"),
    ("code", "# Train Random Forest\nprint(\"Training Random Forest...\")\nn_jobs = -1\nrf = RandomForestClassifier(\n    n_estimators=300,\n    min_samples_leaf=2,\n    class_weight='balanced',\n    random_state=SEED,\n    n_jobs=n_jobs\n)\nrf.fit(X_train, y_train)\nprint(\"Training complete.\")"),
    ("code", "# Evaluate metrics\nprint(\"Evaluating on test split...\")\ny_pred = rf.predict(X_test)\ny_pred_proba = rf.predict_proba(X_test)[:, 1]\n\nacc = accuracy_score(y_test, y_pred)\nmacro_f1 = f1_score(y_test, y_pred, average='macro')\nmalware_f1 = f1_score(y_test, y_pred, pos_label=1)\ncm = confusion_matrix(y_test, y_pred).tolist()\nreport = classification_report(y_test, y_pred, output_dict=True)\n\nmetrics = {\n    'accuracy': acc,\n    'macro_f1': macro_f1,\n    'malware_f1': malware_f1,\n    'confusion_matrix': cm,\n    'classification_report': report\n}\n\nwith open(os.path.join(out_dir, 'metrics.json'), 'w') as f:\n    json.dump(metrics, f, indent=4)\n\nprint(f\"Accuracy: {acc:.4f}\")\nprint(f\"Macro-F1: {macro_f1:.4f}\")\nprint(f\"Malware F1: {malware_f1:.4f}\")"),
    ("code", "# Benchmark inference time\n# Warm-up pass\n_ = rf.predict(X_test)\n\nprint(\"Benchmarking inference time...\")\nsingle_sample = X_test[0:1]\nsingle_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    rf.predict(single_sample)\n    single_times.append(time.perf_counter() - start)\n\nbatch_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    rf.predict(X_test)\n    batch_times.append(time.perf_counter() - start)\n\nsingle_mean, single_std = np.mean(single_times), np.std(single_times)\nbatch_mean, batch_std = np.mean(batch_times), np.std(batch_times)\n\ntiming = {\n    'n_jobs_used': n_jobs,\n    'single_sample_latency_sec': {'mean': single_mean, 'std': single_std},\n    'batch_throughput_sec': {'mean': batch_mean, 'std': batch_std},\n    'batch_size': len(X_test)\n}\n\nwith open(os.path.join(out_dir, 'timing.json'), 'w') as f:\n    json.dump(timing, f, indent=4)\n\nprint(f\"n_jobs used: {n_jobs}\")\nprint(f\"Single-sample latency: {single_mean:.5f}s ± {single_std:.5f}s\")\nprint(f\"Batch throughput: {batch_mean:.5f}s ± {batch_std:.5f}s\")"),
    ("code", "# Measure model size\nmodel_path = os.path.join(out_dir, 'model.joblib')\njoblib.dump(rf, model_path)\nfile_size_bytes = os.path.getsize(model_path)\nfile_size_mb = file_size_bytes / (1024 * 1024)\n\ntotal_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)\navg_depth = np.mean([tree.tree_.max_depth for tree in rf.estimators_])\n\nsize_info = {\n    'file_size_bytes': file_size_bytes,\n    'file_size_mb': file_size_mb,\n    'structural_complexity': {\n        'num_trees': len(rf.estimators_),\n        'average_tree_depth': avg_depth,\n        'total_nodes': total_nodes\n    }\n}\n\nwith open(os.path.join(out_dir, 'size.json'), 'w') as f:\n    json.dump(size_info, f, indent=4)\n\nprint(f\"Model file size: {file_size_mb:.2f} MB\")\nprint(f\"Structural complexity: {len(rf.estimators_)} trees, avg depth {avg_depth:.2f}, {total_nodes} total nodes\")"),
    ("code", "# Save artifacts\npreds_df = pd.DataFrame({\n    'y_true': y_test,\n    'y_pred': y_pred,\n    'y_pred_proba': y_pred_proba\n})\npreds_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)\nprint(\"All artifacts saved successfully.\")")
]
write_notebook("rf_pipeline.ipynb", rf_cells)

# ----------------- XGB Notebook ----------------- #
xgb_cells = [
    ("markdown", "# XGBoost Pipeline\nImplements the XGBoost baseline as per `pipeline_design.txt` and prompt instructions."),
    ("code", "import pandas as pd\nimport numpy as np\nimport json\nimport os\nimport joblib\nimport time\nimport sklearn\nimport xgboost as xgb\nfrom sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\nimport sys\n\nprint(f\"Python: {sys.version}\")\nprint(f\"pandas: {pd.__version__}\")\nprint(f\"xgboost: {xgb.__version__}\")"),
    ("code", "sys.path.append('.')\nfrom config import SEED\nprint(f\"Shared SEED: {SEED}\")"),
    ("code", "out_dir = 'outputs/xgboost'\nos.makedirs(out_dir, exist_ok=True)"),
    ("code", "# Load data and splits\nprint(\"Loading data...\")\ndf = pd.read_parquet('cleaned_dataset.parquet')\nfeature_cols = [c for c in df.columns if c != 'label']\n\nwith open('split_indices.json', 'r') as f:\n    splits = json.load(f)\n\nX_train = df.loc[splits['train_idx'], feature_cols].values\ny_train = df.loc[splits['train_idx'], 'label'].values\nX_val = df.loc[splits['val_idx'], feature_cols].values\ny_val = df.loc[splits['val_idx'], 'label'].values\nX_test = df.loc[splits['test_idx'], feature_cols].values\ny_test = df.loc[splits['test_idx'], 'label'].values"),
    ("code", "# Preprocessing\n# No preprocessing needed for XGBoost (no scaling)\nscaler = None\njoblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))\nprint(\"Null scaler saved.\")"),
    ("code", "# Train XGBoost\nnum_neg = (y_train == 0).sum()\nnum_pos = (y_train == 1).sum()\nscale_pos_weight = num_neg / num_pos\nprint(f\"scale_pos_weight: {scale_pos_weight:.4f}\")\n\nxgb_model = xgb.XGBClassifier(\n    n_estimators=300,\n    max_depth=6,\n    learning_rate=0.1,\n    scale_pos_weight=scale_pos_weight,\n    subsample=0.8,\n    colsample_bytree=0.8,\n    eval_metric='logloss',\n    early_stopping_rounds=10,\n    random_state=SEED,\n    n_jobs=-1\n)\n\nprint(\"Training XGBoost...\")\nxgb_model.fit(\n    X_train, y_train,\n    eval_set=[(X_val, y_val)],\n    verbose=False\n)\nprint(\"Training complete.\")"),
    ("code", "# Evaluate metrics\nprint(\"Evaluating on test split...\")\ny_pred = xgb_model.predict(X_test)\ny_pred_proba = xgb_model.predict_proba(X_test)[:, 1]\n\nacc = accuracy_score(y_test, y_pred)\nmacro_f1 = f1_score(y_test, y_pred, average='macro')\nmalware_f1 = f1_score(y_test, y_pred, pos_label=1)\ncm = confusion_matrix(y_test, y_pred).tolist()\nreport = classification_report(y_test, y_pred, output_dict=True)\n\nmetrics = {\n    'accuracy': acc,\n    'macro_f1': macro_f1,\n    'malware_f1': malware_f1,\n    'confusion_matrix': cm,\n    'classification_report': report\n}\nwith open(os.path.join(out_dir, 'metrics.json'), 'w') as f:\n    json.dump(metrics, f, indent=4)"),
    ("code", "# Benchmark inference time\n_ = xgb_model.predict(X_test)\n\nprint(\"Benchmarking inference time...\")\nsingle_sample = X_test[0:1]\nsingle_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    xgb_model.predict(single_sample)\n    single_times.append(time.perf_counter() - start)\n\nbatch_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    xgb_model.predict(X_test)\n    batch_times.append(time.perf_counter() - start)\n\ntiming = {\n    'n_jobs_used': -1,\n    'single_sample_latency_sec': {'mean': np.mean(single_times), 'std': np.std(single_times)},\n    'batch_throughput_sec': {'mean': np.mean(batch_times), 'std': np.std(batch_times)},\n    'batch_size': len(X_test)\n}\nwith open(os.path.join(out_dir, 'timing.json'), 'w') as f:\n    json.dump(timing, f, indent=4)"),
    ("code", "# Measure model size\nmodel_path = os.path.join(out_dir, 'model.json')\nxgb_model.save_model(model_path)\nfile_size_bytes = os.path.getsize(model_path)\n\ndump = xgb_model.get_booster().get_dump()\nnum_trees = len(dump)\ntotal_nodes = sum(tree.count('\\n') for tree in dump)\n\nsize_info = {\n    'file_size_bytes': file_size_bytes,\n    'file_size_mb': file_size_bytes / (1024 * 1024),\n    'structural_complexity': {\n        'num_trees': num_trees,\n        'total_nodes_approx': total_nodes\n    }\n}\nwith open(os.path.join(out_dir, 'size.json'), 'w') as f:\n    json.dump(size_info, f, indent=4)"),
    ("code", "# Save artifacts\npreds_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba})\npreds_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)\nprint(\"All artifacts saved successfully.\")")
]
write_notebook("xgboost_pipeline.ipynb", xgb_cells)

# ----------------- NN Notebook ----------------- #
nn_cells = [
    ("markdown", "# Neural Network Pipeline\nImplements the Neural Network baseline as per `pipeline_design.txt` and prompt instructions."),
    ("code", "import pandas as pd\nimport numpy as np\nimport json\nimport os\nimport joblib\nimport time\nimport sklearn\nfrom sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\nfrom sklearn.preprocessing import StandardScaler\nimport tensorflow as tf\nfrom tensorflow.keras.layers import Input, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping\nfrom tensorflow.keras.optimizers import Adam\nimport sys\n\nprint(f\"Python: {sys.version}\")\nprint(f\"TensorFlow: {tf.__version__}\")"),
    ("code", "sys.path.append('.')\nfrom config import SEED\nprint(f\"Shared SEED: {SEED}\")\ntf.random.set_seed(SEED)\nnp.random.seed(SEED)"),
    ("code", "out_dir = 'outputs/nn'\nos.makedirs(out_dir, exist_ok=True)"),
    ("code", "# Load data and splits\nprint(\"Loading data...\")\ndf = pd.read_parquet('cleaned_dataset.parquet')\nfeature_cols = [c for c in df.columns if c != 'label']\n\nwith open('split_indices.json', 'r') as f:\n    splits = json.load(f)\n\nX_train = df.loc[splits['train_idx'], feature_cols].values\ny_train = df.loc[splits['train_idx'], 'label'].values\nX_val = df.loc[splits['val_idx'], feature_cols].values\ny_val = df.loc[splits['val_idx'], 'label'].values\nX_test = df.loc[splits['test_idx'], feature_cols].values\ny_test = df.loc[splits['test_idx'], 'label'].values"),
    ("code", "# Preprocessing: log1p then StandardScaler\nprint(\"Applying log1p transform...\")\nX_train_log = np.log1p(X_train)\nX_val_log = np.log1p(X_val)\nX_test_log = np.log1p(X_test)\n\nprint(\"Fitting StandardScaler on training split...\")\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train_log)\nX_val_scaled = scaler.transform(X_val_log)\nX_test_scaled = scaler.transform(X_test_log)\n\njoblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))\nprint(\"Scaler saved.\")"),
    ("code", "# Train Neural Network\nnum_neg = (y_train == 0).sum()\nnum_pos = (y_train == 1).sum()\nweight_for_0 = (1 / num_neg) * (len(y_train) / 2.0)\nweight_for_1 = (1 / num_pos) * (len(y_train) / 2.0)\nclass_weight = {0: weight_for_0, 1: weight_for_1}\nprint(f\"Class weights: {class_weight}\")\n\ninputs = Input(shape=(X_train_scaled.shape[1],))\nx = Dense(128, activation='relu')(inputs)\nx = Dropout(0.4)(x)\nx = Dense(64, activation='relu')(x)\nx = Dropout(0.3)(x)\noutputs = Dense(1, activation='sigmoid')(x)\n\nmodel = Model(inputs=inputs, outputs=outputs)\nmodel.compile(\n    optimizer=Adam(learning_rate=1e-3),\n    loss='binary_crossentropy',\n    metrics=['accuracy']\n)\n\nmodel.summary()\ntotal_params = model.count_params()\nassert total_params < 500000, f\"Parameter count {total_params} exceeds 500K!\"\n\ncallbacks = [\n    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),\n    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)\n]\n\nprint(\"Training NN...\")\nhistory = model.fit(\n    X_train_scaled, y_train,\n    validation_data=(X_val_scaled, y_val),\n    epochs=100,\n    batch_size=64,\n    class_weight=class_weight,\n    callbacks=callbacks,\n    verbose=1\n)\nprint(\"Training complete.\")"),
    ("code", "# Evaluate metrics\nprint(\"Evaluating on test split...\")\ny_pred_proba = model.predict(X_test_scaled).flatten()\ny_pred = (y_pred_proba >= 0.5).astype(int)\n\nacc = accuracy_score(y_test, y_pred)\nmacro_f1 = f1_score(y_test, y_pred, average='macro')\nmalware_f1 = f1_score(y_test, y_pred, pos_label=1)\ncm = confusion_matrix(y_test, y_pred).tolist()\nreport = classification_report(y_test, y_pred, output_dict=True)\n\nmetrics = {\n    'accuracy': acc,\n    'macro_f1': macro_f1,\n    'malware_f1': malware_f1,\n    'confusion_matrix': cm,\n    'classification_report': report\n}\nwith open(os.path.join(out_dir, 'metrics.json'), 'w') as f:\n    json.dump(metrics, f, indent=4)"),
    ("code", "# Benchmark inference time\n_ = model.predict(X_test_scaled, batch_size=128, verbose=0)\n\nprint(\"Benchmarking inference time...\")\nsingle_sample = X_test_scaled[0:1]\nsingle_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    model.predict(single_sample, verbose=0)\n    single_times.append(time.perf_counter() - start)\n\nbatch_times = []\nfor _ in range(100):\n    start = time.perf_counter()\n    model.predict(X_test_scaled, batch_size=128, verbose=0)\n    batch_times.append(time.perf_counter() - start)\n\nthreads = tf.config.threading.get_intra_op_parallelism_threads()\ntiming = {\n    'framework_threads': threads,\n    'single_sample_latency_sec': {'mean': np.mean(single_times), 'std': np.std(single_times)},\n    'batch_throughput_sec': {'mean': np.mean(batch_times), 'std': np.std(batch_times)},\n    'batch_size': len(X_test_scaled)\n}\nwith open(os.path.join(out_dir, 'timing.json'), 'w') as f:\n    json.dump(timing, f, indent=4)\nprint(f\"Framework threads: {threads}\")"),
    ("code", "# Measure model size\nmodel_path = os.path.join(out_dir, 'model.keras')\nmodel.save(model_path)\nfile_size_bytes = os.path.getsize(model_path)\n\nsize_info = {\n    'file_size_bytes': file_size_bytes,\n    'file_size_mb': file_size_bytes / (1024 * 1024),\n    'structural_complexity': {\n        'total_params': total_params\n    }\n}\nwith open(os.path.join(out_dir, 'size.json'), 'w') as f:\n    json.dump(size_info, f, indent=4)"),
    ("code", "# Save artifacts\npreds_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba})\npreds_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)\nprint(\"All artifacts saved successfully.\")")
]
write_notebook("nn_pipeline.ipynb", nn_cells)

print("Notebook generation completed.")
