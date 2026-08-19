#!/usr/bin/env python3
"""
Aero-Sense: 1D-CNN + BiLSTM Deep Maneuver Classifier Training & ONNX Export
Trains hybrid PyTorch model on 30x8 kinematic sliding windows, computes F1 metrics,
exports to ONNX format (opset 17), and verifies numerical parity with ONNX Runtime.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr:
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort


CLASS_NAMES = [
    "Straight Cruise",
    "Coordinated Turn",
    "Climb",
    "Descent / Dive",
    "Orbit / Holding",
    "Evasive Maneuver",
]


class ManeuverCNNBiLSTM(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional LSTM Architecture for Kinematic Maneuver Classification.
    Extracts local temporal shock/impulse features (CNN) and long-range sequence context (BiLSTM).
    """

    def __init__(self, in_features: int = 8, seq_len: int = 30, num_classes: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = in_features

        # 1D-CNN Feature Extractor: processes shape (Batch, in_features, seq_len)
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Bidirectional LSTM: processes shape (Batch, seq_len, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Classification Head (BiLSTM output: 64 * 2 = 128)
        self.fc1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (Batch, seq_len, in_features) e.g., (Batch, 30, 8)
        Output: Logits of shape (Batch, num_classes)
        """
        # Transpose to (Batch, in_features, seq_len) for Conv1D
        x_conv = x.transpose(1, 2)
        h = self.relu1(self.bn1(self.conv1(x_conv)))
        h = self.relu2(self.bn2(self.conv2(h)))

        # Transpose back to (Batch, seq_len, 128) for LSTM
        h_lstm = h.transpose(1, 2)
        lstm_out, _ = self.lstm(h_lstm)

        # Global average pooling over time steps for robust temporal representation
        pooled = torch.mean(lstm_out, dim=1)  # Shape: (Batch, 128)

        # Dense Classifier
        out = self.fc_out(self.dropout(self.relu3(self.fc1(pooled))))
        return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> Dict[str, float]:
    """Computes class-based precision, recall, F1, accuracy, and macro F1."""
    accuracy = np.mean(y_true == y_pred)
    f1_scores = []
    
    print("\n" + "=" * 65)
    print(f"{'Class ID':<10} {'Maneuver Name':<22} {'Precision':<11} {'Recall':<11} {'F1-Score':<10}")
    print("-" * 65)

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{c:<10} {CLASS_NAMES[c]:<22} {prec:<11.4f} {rec:<11.4f} {f1:<10.4f}")

    macro_f1 = float(np.mean(f1_scores))
    print("-" * 65)
    print(f"Overall Accuracy: {accuracy * 100:.2f}% | Macro F1-Score: {macro_f1 * 100:.2f}%")
    print("=" * 65)

    return {"accuracy": float(accuracy), "macro_f1": macro_f1}


def train_and_export(
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    """Executes full training loop, computes evaluation metrics, and exports ONNX model."""
    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    # Load normalized datasets
    train_X = np.load(data_dir / "train_data.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    val_X = np.load(data_dir / "val_data.npy")
    val_y = np.load(data_dir / "val_labels.npy")

    train_dataset = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long())
    val_dataset = TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ManeuverCNNBiLSTM(in_features=8, seq_len=30, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_macro_f1 = 0.0
    best_weights_path = models_dir / "model_best.pt"

    print("\n[*] Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        avg_train_loss = total_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc = np.mean(val_preds == val_targets)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc * 100:.2f}%")

    # Final Comprehensive Metrics
    print("\n[*] Evaluating Best Model Metrics on Validation Set:")
    metrics = compute_metrics(val_targets, val_preds)

    # Save PyTorch weights
    torch.save(model.state_dict(), best_weights_path)
    print(f"[+] Saved PyTorch weights to: {best_weights_path}")

    # ================= ONNX EXPORT =================
    onnx_path = models_dir / "model_cnn_lstm.onnx"
    print(f"\n[*] Exporting to ONNX model: {onnx_path}...")

    model.eval().to("cpu")
    dummy_input = torch.randn(1, 30, 8, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["telemetry_window"],
            output_names=["maneuver_logits"],
            dynamic_axes={
                "telemetry_window": {0: "batch_size"},
                "maneuver_logits": {0: "batch_size"},
            },
            dynamo=False,
        )
    except TypeError:
        # For older PyTorch versions without dynamo argument
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["telemetry_window"],
            output_names=["maneuver_logits"],
            dynamic_axes={
                "telemetry_window": {0: "batch_size"},
                "maneuver_logits": {0: "batch_size"},
            },
        )

    # Verify ONNX model integrity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model integrity verified successfully!")

    # Verify Numerical Parity between PyTorch and ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"telemetry_window": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        py_output = model(dummy_input).numpy()

    mse_diff = float(np.mean((py_output - ort_outputs[0]) ** 2))
    print(f"[+] Numerical Parity Check (PyTorch vs ONNX Runtime) -> MSE: {mse_diff:.2e}")
    assert mse_diff < 1e-5, f"Numerical parity failed! MSE = {mse_diff}"
    print("[+] Cross-validation passed! Model ready for C++ Core Engine deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aero-Sense 1D-CNN + BiLSTM Training & ONNX Export")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed .npy files")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_and_export(
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
