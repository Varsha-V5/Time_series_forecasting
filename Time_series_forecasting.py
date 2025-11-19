"""
Advanced Time Series Forecasting with Attention Mechanism (Transformer) + LSTM baseline
Single-file project implementing:
- synthetic multivariate dataset generation (2000 daily obs) + save CSV
- EDA summary prints (no plots for headless runs) and optional plots saved
- preprocessing (train/val/test split, scaling), sliding-window sequence generation
- Transformer encoder-decoder forecasting model (seq2seq) with positional encoding
- Baseline LSTM model (seq2one or seq2seq version)
- Training loops with AdamW, CosineAnnealingLR, dropout schedule, checkpointing
- Evaluation metrics: RMSE, MAE, MAPE
- Prediction plots saved to ./outputs/
- Permutation importance for interpretability
- Saves: dataset CSV, model checkpoints, plots
Run: python transformer_forecasting_project.py
"""

import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

# If you uploaded reference images, path available here:
UPLOADED_FILE_PATH = "/mnt/data/Screenshot 2025-11-19 175851.png"

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PATH = Path("./transformer_timeseries_dataset.csv")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# 1) DATA GENERATION
# -----------------------
def generate_dataset(n=2000, seed=SEED, save_path=DATA_PATH):
    np.random.seed(seed)
    date_index = pd.date_range(start="2000-01-01", periods=n, freq="D")
    t = np.arange(n)

    # Components
    trend = 0.0008 * t
    seasonal_yearly = 2.5 * np.sin(2 * np.pi * t / 365.25)
    seasonal_weekly = 0.8 * np.sin(2 * np.pi * t / 7.0)
    ar = np.zeros(n)
    phi = [0.5, -0.25]
    for i in range(2, n):
        ar[i] = phi[0] * ar[i-1] + phi[1] * ar[i-2]

    exog1 = 0.9 * np.sin(2 * np.pi * t / 90.0) + 0.2 * np.random.normal(size=n)
    exog2 = np.cos(2 * np.pi * t / 30.0) + 0.15 * np.random.normal(size=n)
    exog3 = np.random.normal(scale=0.6, size=n)

    noise = np.random.normal(scale=0.9, size=n)
    target = 4.0 + trend + seasonal_yearly + seasonal_weekly + 1.3*ar + 0.7*exog1 - 0.5*exog2 + 0.25*exog3 + noise

    max_lag = 14
    lags = {f"lag_{lag}": pd.Series(target).shift(lag).fillna(method="bfill").values for lag in range(1, max_lag+1)}

    df = pd.DataFrame({
        "timestamp": date_index,
        "target": target,
        "exog1": exog1,
        "exog2": exog2,
        "exog3": exog3,
        "trend": trend,
        "seasonal_yearly": seasonal_yearly,
        "seasonal_weekly": seasonal_weekly
    })
    for k, v in lags.items():
        df[k] = v

    df.to_csv(save_path, index=False)
    print(f"[DATA] Saved dataset to {save_path}")
    return df

# -----------------------
# 2) SEQUENCE DATASET (seq2seq for transformer)
# -----------------------
class Sequence2SeqDataset(Dataset):
    """
    Prepares sequences for seq2seq forecasting:
    - encoder_input: last enc_len timesteps
    - decoder_target: next dec_len timesteps (used as teacher forcing during training or for evaluation)
    - returns (enc_in, dec_in, dec_out) where dec_in is decoder inputs (shifted)
    """
    def __init__(self, data_df, feature_cols, target_col='target', enc_len=96, dec_len=24):
        self.enc_len = enc_len
        self.dec_len = dec_len
        X = data_df[feature_cols].values.astype(np.float32)
        y = data_df[target_col].values.astype(np.float32)
        self.X = X
        self.y = y
        self.n = len(y)

    def __len__(self):
        return max(0, self.n - self.enc_len - self.dec_len + 1)

    def __getitem__(self, idx):
        enc_in = self.X[idx: idx + self.enc_len]  # (enc_len, n_features)
        dec_out = self.y[idx + self.enc_len: idx + self.enc_len + self.dec_len]  # (dec_len,)
        # decoder input for teacher forcing: previous outputs; here use last value repeated or zeros — simple approach
        # dec_in shape: (dec_len, n_features_target) -> for univariate target we shape (dec_len, 1)
        dec_in = np.concatenate([np.array([self.y[idx + self.enc_len - 1]]), dec_out[:-1]])
        dec_in = dec_in.reshape(-1, 1).astype(np.float32)
        return enc_in, dec_in, dec_out.astype(np.float32)

# -----------------------
# 3) Positional Encoding (standard)
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# -----------------------
# 4) Transformer Seq2Seq Model (simple, clear)
# -----------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=2,
                 dim_feedforward=128, dropout=0.1, dec_out_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # Embedding for encoder inputs (project features -> d_model)
        self.enc_embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=2000)
        # decoder embedding: for previous target steps (univariate) -> d_model
        self.dec_embed = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.out_layer = nn.Linear(d_model, dec_out_dim)

    def forward(self, enc_in, dec_in):
        # enc_in: (batch, enc_len, input_dim)
        # dec_in: (batch, dec_len, 1)
        enc = self.enc_embed(enc_in) * math.sqrt(self.d_model)
        enc = self.pos_enc(enc)
        dec = self.dec_embed(dec_in) * math.sqrt(self.d_model)
        dec = self.pos_enc(dec)
        # source_mask/target_mask can be added for causal decoding; use default masks for now
        # transformer expects (batch, seq, feature) with batch_first=True
        out = self.transformer(src=enc, tgt=dec)  # (batch, dec_len, d_model)
        out = self.out_layer(out)  # (batch, dec_len, 1)
        return out.squeeze(-1)  # (batch, dec_len)

# -----------------------
# 5) Baseline LSTM (seq2seq-ish or seq2one)
# -----------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.2, dec_len=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, dec_len)
        )
    def forward(self, x):
        # x: (batch, enc_len, input_dim) -> we predict dec_len outputs
        out, _ = self.lstm(x)
        # use last hidden state
        last = out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last)  # (batch, dec_len)
        return out

# -----------------------
# 6) Training & Utility functions
# -----------------------
def split_scaler(df, feature_cols, test_size=0.15, val_size=0.15):
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_df = df.iloc[: n - test_n - val_n].reset_index(drop=True)
    val_df = df.iloc[n - test_n - val_n: n - test_n].reset_index(drop=True)
    test_df = df.iloc[n - test_n:].reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    for d in [train_df, val_df, test_df]:
        d[feature_cols] = scaler.transform(d[feature_cols])
    return train_df, val_df, test_df, scaler

def rmse(a, b):
    return math.sqrt(mean_squared_error(a, b))

def mape(a, b):
    return np.mean(np.abs((a - b) / (a + 1e-8))) * 100

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for enc_in, dec_in, dec_out in loader:
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        dec_out = dec_out.to(device)
        optimizer.zero_grad()
        preds = model(enc_in, dec_in) if isinstance(model, TimeSeriesTransformer) else model(enc_in)
        loss = loss_fn(preds, dec_out)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

def eval_model(model, loader, device):
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for enc_in, dec_in, dec_out in loader:
            enc_in = enc_in.to(device)
            dec_in = dec_in.to(device)
            dec_out = dec_out.to(device)
            out = model(enc_in, dec_in) if isinstance(model, TimeSeriesTransformer) else model(enc_in)
            preds_list.append(out.detach().cpu().numpy())
            trues_list.append(dec_out.detach().cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    return preds, trues

# -----------------------
# 7) Permutation importance for seq features
# -----------------------
def permutation_importance_seq(model, X_enc, dec_in, y_true, feat_idx, device, batches=64):
    # X_enc: numpy (n_samples, enc_len, n_features)
    baseline_preds = model(torch.tensor(X_enc, dtype=torch.float32).to(device),
                           torch.tensor(dec_in, dtype=torch.float32).to(device)).detach().cpu().numpy()
    base_rmse = rmse(y_true.flatten(), baseline_preds.flatten())
    Xp = X_enc.copy()
    # permute feature across samples for all timesteps
    Xp[:, :, feat_idx] = np.random.permutation(Xp[:, :, feat_idx])
    perm_preds = model(torch.tensor(Xp, dtype=torch.float32).to(device),
                       torch.tensor(dec_in, dtype=torch.float32).to(device)).detach().cpu().numpy()
    perm_rmse = rmse(y_true.flatten(), perm_preds.flatten())
    return perm_rmse - base_rmse, base_rmse, perm_rmse

# -----------------------
# 8) Plotting utility
# -----------------------
def save_prediction_plot(trues, preds, save_path, num_points=200, title="Actual vs Predicted"):
    plt.figure(figsize=(12,4))
    plt.plot(trues[:num_points].flatten(), label="Actual", linewidth=1)
    plt.plot(preds[:num_points].flatten(), label="Predicted", linewidth=1)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------
# 9) Full pipeline
# -----------------------
def run_pipeline(csv_path=DATA_PATH, enc_len=96, dec_len=24, batch_size=32, epochs=40, device=DEVICE, use_saved=False):
    # 1) Load or generate dataset
    if not csv_path.exists() or not use_saved:
        df = generate_dataset(save_path=csv_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"[DATA] Loaded {csv_path}")

    feature_cols = [c for c in df.columns if c not in ['timestamp', 'target']]
    print("[EDA] dataset shape:", df.shape)
    print(df.describe().T.loc[['mean','std']])

    # 2) Preprocess and split
    train_df, val_df, test_df, scaler = split_scaler(df, feature_cols, test_size=0.15, val_size=0.15)
    print(f"[SPLIT] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Instantiate datasets
    train_ds = Sequence2SeqDataset(train_df, feature_cols, enc_len=enc_len, dec_len=dec_len)
    val_ds = Sequence2SeqDataset(val_df, feature_cols, enc_len=enc_len, dec_len=dec_len)
    test_ds = Sequence2SeqDataset(test_df, feature_cols, enc_len=enc_len, dec_len=dec_len)
    print(f"[DATASET] train samples: {len(train_ds)}, val samples: {len(val_ds)}, test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3) Create models
    n_features = len(feature_cols)
    transformer = TimeSeriesTransformer(input_dim=n_features, d_model=64, nhead=4,
                                        num_encoder_layers=3, num_decoder_layers=2, dropout=0.1, dim_feedforward=128).to(device)
    lstm = LSTMForecast(input_dim=n_features, hidden_size=128, num_layers=2, dropout=0.2, dec_len=dec_len).to(device)

    # 4) Optimizers and schedulers
    t_opt = optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1e-4)
    t_sched = optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=epochs)
    l_opt = optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
    l_sched = optim.lr_scheduler.CosineAnnealingLR(l_opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    # dropout schedule (example: linearly increase between 0.05 to 0.4 across epochs)
    def dropout_schedule(epoch, model):
        p = min(0.4, 0.05 + (epoch / max(1, epochs)) * 0.35)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    # 5) Training loop (train both models)
    best_val_rmse = {"transformer": 1e9, "lstm": 1e9}
    for epoch in range(1, epochs+1):
        # dropout schedule
        dropout_schedule(epoch, transformer)
        dropout_schedule(epoch, lstm)

        t_loss = train_epoch(transformer, train_loader, t_opt, loss_fn, device)
        l_loss = train_epoch(lstm, train_loader, l_opt, loss_fn, device)

        t_sched.step(); l_sched.step()

        t_preds_val, t_trues_val = eval_model(transformer, val_loader, device)
        l_preds_val, l_trues_val = eval_model(lstm, val_loader, device)

        t_val_rmse = rmse(t_trues_val.flatten(), t_preds_val.flatten())
        l_val_rmse = rmse(l_trues_val.flatten(), l_preds_val.flatten())

        if t_val_rmse < best_val_rmse["transformer"]:
            best_val_rmse["transformer"] = t_val_rmse
            torch.save(transformer.state_dict(), OUTPUT_DIR / "transformer_best.pth")
        if l_val_rmse < best_val_rmse["lstm"]:
            best_val_rmse["lstm"] = l_val_rmse
            torch.save(lstm.state_dict(), OUTPUT_DIR / "lstm_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch}/{epochs}] T_loss={t_loss:.4f} T_val_rmse={t_val_rmse:.4f} | L_loss={l_loss:.4f} L_val_rmse={l_val_rmse:.4f}")

    # 6) Final evaluation on test set (load best models)
    transformer.load_state_dict(torch.load(OUTPUT_DIR / "transformer_best.pth"))
    lstm.load_state_dict(torch.load(OUTPUT_DIR / "lstm_best.pth"))

    t_preds_test, t_trues_test = eval_model(transformer, test_loader, device)
    l_preds_test, l_trues_test = eval_model(lstm, test_loader, device)

    t_rmse = rmse(t_trues_test.flatten(), t_preds_test.flatten())
    t_mae = mean_absolute_error(t_trues_test.flatten(), t_preds_test.flatten())
    t_mape = mape(t_trues_test.flatten(), t_preds_test.flatten())

    l_rmse = rmse(l_trues_test.flatten(), l_preds_test.flatten())
    l_mae = mean_absolute_error(l_trues_test.flatten(), l_preds_test.flatten())
    l_mape = mape(l_trues_test.flatten(), l_preds_test.flatten())

    print("\n[RESULTS] Transformer Test RMSE:{:.4f} MAE:{:.4f} MAPE:{:.2f}%".format(t_rmse, t_mae, t_mape))
    print("[RESULTS] LSTM Test RMSE:{:.4f} MAE:{:.4f} MAPE:{:.2f}%".format(l_rmse, l_mae, l_mape))

    # Save sample prediction plots (flatten first dec_len predictions to show series)
    # We will flatten to a 1-D series by concatenating horizon steps for visualization.
    save_prediction_plot(t_trues_test.flatten(), t_preds_test.flatten(), OUTPUT_DIR / "transformer_preds.png", num_points=500, title="Transformer: Actual vs Predicted (test)")
    save_prediction_plot(l_trues_test.flatten(), l_preds_test.flatten(), OUTPUT_DIR / "lstm_preds.png", num_points=500, title="LSTM: Actual vs Predicted (test)")
    # Overlay comparison
    plt.figure(figsize=(12,4))
    plt.plot(t_trues_test.flatten()[:500], label="Actual")
    plt.plot(t_preds_test.flatten()[:500], label="Transformer")
    plt.plot(l_preds_test.flatten()[:500], label="LSTM", alpha=0.7)
    plt.legend(); plt.title("Actual vs Transformer vs LSTM (test segment)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparison_overlay.png"); plt.close()

    # 7) Permutation importance for a subset of features (first window of test set)
    # Build arrays for permutation function: collect all enc inputs + dec_in and y_true
    enc_list, dec_in_list, dec_out_list = [], [], []
    for enc_in, dec_in, dec_out in test_loader:
        enc_list.append(enc_in.numpy()); dec_in_list.append(dec_in.numpy()); dec_out_list.append(dec_out.numpy())
    X_enc = np.concatenate(enc_list, axis=0)
    dec_in_arr = np.concatenate(dec_in_list, axis=0)
    y_true_arr = np.concatenate(dec_out_list, axis=0)
    importances = {}
    for fi in range(X_enc.shape[2]):  # features
        imp, base, perm = permutation_importance_seq(transformer, X_enc, dec_in_arr, y_true_arr, feat_idx=fi, device=device)
        importances[feature_cols[fi]] = imp
    sorted_imp = sorted(importances.items(), key=lambda x: -abs(x[1]))
    print("\n[Permutation importances] (feature -> delta RMSE):")
    for k, v in sorted_imp:
        print(f"  {k}: {v:.4f}")

    # Save results CSV summary
    summary = {
        "transformer_rmse": t_rmse, "transformer_mae": t_mae, "transformer_mape": t_mape,
        "lstm_rmse": l_rmse, "lstm_mae": l_mae, "lstm_mape": l_mape
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / "results_summary.csv")
    print(f"[DONE] Outputs and plots saved to {OUTPUT_DIR}")

    return {
        "transformer": {"rmse": t_rmse, "mae": t_mae, "mape": t_mape},
        "lstm": {"rmse": l_rmse, "mae": l_mae, "mape": l_mape},
        "feature_importances": importances
    }

# -----------------------
# 10) ENTRY POINT
# -----------------------
if __name__ == "__main__":
    results = run_pipeline(csv_path=DATA_PATH, enc_len=96, dec_len=24, batch_size=64, epochs=30, device=DEVICE)
    print(results)
