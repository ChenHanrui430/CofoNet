import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc, cohen_kappa_score
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightningModule
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR

# ==============================================================================
# 1. Configuration 
# ==============================================================================
# Use relative paths for portability
BASE_PATH = './data/' 
OUTPUT_DIR = './results'

SHAPES = {
    'sirinam_wf': 5000, 
    'sirinam_vf': 25000, 
    'rahman': 25000, 
    'hayden': 25000,
    'schuster8': 480, 
    'SALT': 25000  
} 

# Best Hyperparameters Dictionary
BEST_HYPERPARAMETERS = {
    # CoFoNet + SALT (Ours)
    "SALT_tor_youtube":{
        "filters": 48,
        "drop_path_rate": 0.35,
        "weight_decay": 0.022,
        "lr": 0.0024,
        "fc_dropout": 0.6,
        "batch_size": 112,
        "factor": 0.51
    },
}

class MemmapDataset(Dataset):
    def __init__(self, file_prefix, representation, data_points, target_label_name, in_memory=False, augment=False):
        super().__init__()
        
        feat_path = f"{file_prefix}_features.npy"
        label_path = f"{file_prefix}_label_{target_label_name}.npy"
        play_label_path = f"{file_prefix}_label_playlist_code.npy"
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing: {feat_path}")

        if in_memory:
            print(f" [Dataset] Loading {feat_path} into RAM...")
            self.X = np.load(feat_path)
            self.y_vid = np.load(label_path)
            self.y_play = np.load(play_label_path)
        else:
            self.X = np.load(feat_path, mmap_mode='r')
            self.y_vid = np.load(label_path, mmap_mode='r')
            self.y_play = np.load(play_label_path, mmap_mode='r')
            
        self.data_points = data_points
        self.representation = representation
        self.augment = augment
        
        self.mode = 'default'
        if 'schuster' in representation: self.mode = 'schuster'
        elif 'rschuster' in representation: self.mode = 'rschuster'
        self.is_2d = (representation == 'SALT' or 'dschuster' in representation)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        raw_view = self.X[idx].copy()
        label_vid = self.y_vid[idx]
        label_play = self.y_play[idx]

        t_vec = torch.tensor(raw_view, dtype=torch.float32)

        if self.is_2d:
            t_vec = t_vec.view(self.data_points, 2)
        else:
            t_vec = t_vec.view(self.data_points, 1)

        if self.mode == 'schuster':
            final_data = t_vec[:, 0].unsqueeze(0)
        elif self.mode == 'rschuster':
            final_data = t_vec[:, 1].unsqueeze(0)
        else:
            final_data = t_vec.transpose(0, 1)
            
        return final_data, label_vid, label_play


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_p = F.log_softmax(logits, dim=-1)
            ce_loss = -torch.sum(true_dist * log_p, dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if isinstance(samples_per_class, torch.Tensor):
            samples_per_class = samples_per_class.cpu().numpy()
            
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AttentionStatisticsPooling(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = x - mean.unsqueeze(2)
        variance = torch.sum(alpha * residuals.pow(2), dim=2)
        std = torch.sqrt(variance.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class FastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, kernel_size=31, drop_path=0.0):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expansion)
        
        layers = []
        if expansion != 1:
            layers.append(nn.Conv1d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))
            
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))
        
        se_hidden = max(hidden_dim // 4, 16) 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, se_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(se_hidden, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        layers.append(nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        self.conv = nn.Sequential(*layers[:-2]) 
        self.project = nn.Sequential(*layers[-2:])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = x * self.se(x) 
        x = self.project(x)
        
        if self.use_res_connect:
            return res + self.drop_path(x)
        return x

class MultiTaskTrafficNet(LightningModule):
    def __init__(self, in_channels, input_length, num_classes_vid, num_classes_play, 
                 samples_per_class_play, hype, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = float(hype.get('weight_decay', 1e-2))
        self.base_filters = int(hype.get('filters', 48))
        self.drop_path_rate = float(hype.get('drop_path_rate', 0.2))
        self.factor = float(hype.get('factor', 0.3))

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, self.base_filters, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(self.base_filters),
            nn.SiLU(inplace=True)
        )

        self.stages = nn.ModuleList()
        stage_configs = [
            [self.base_filters, 2, 2, 11], 
            [self.base_filters*2, 2, 2, 15],
            [self.base_filters*4, 4, 2, 25], 
            [self.base_filters*6, 2, 2, 31]
        ]
        
        in_ch = self.base_filters
        blk_idx = 0
        total_blocks = sum([c[1] for c in stage_configs])

        for out_ch, num_b, stride, k in stage_configs:
            blocks = []
            for i in range(num_b):
                s = stride if i == 0 else 1
                # Stochastic Depth Decay
                dp = self.drop_path_rate * blk_idx / total_blocks
                blocks.append(FastBlock(in_ch, out_ch, s, 2, k, dp))
                in_ch = out_ch
                blk_idx += 1
            self.stages.append(nn.Sequential(*blocks))
        
        dim_stage3 = self.base_filters * 4
        dim_stage4 = self.base_filters * 6
        self.pool_l3 = AttentionStatisticsPooling(dim_stage3, hidden_dim=64)
        self.pool_l4 = AttentionStatisticsPooling(dim_stage4, hidden_dim=64)
        
        fusion_dim = (dim_stage3 * 2) + (dim_stage4 * 2)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            nn.BatchNorm1d(768),
            nn.Mish(inplace=True),
            nn.Dropout(0.5)
        )

        self.head_vid = nn.Linear(768, num_classes_vid)
        self.head_play = nn.Linear(768, num_classes_play)
        self.criterion_vid = FocalLoss(num_classes=num_classes_vid, gamma=2.0, label_smoothing=0.1)
        
        self.criterion_play = ClassBalancedFocalLoss(
            num_classes=num_classes_play,
            samples_per_class=samples_per_class_play,
            gamma=2.0,
            beta=0.999
        )
        
        # --- Metrics ---
        metrics_params = {"task": "multiclass", "num_classes": num_classes_vid}
        self.train_acc_vid = torchmetrics.Accuracy(**metrics_params)
        self.train_f1_vid = torchmetrics.F1Score(**metrics_params, average='macro')
        self.train_f1_play = torchmetrics.F1Score(task="multiclass", num_classes=num_classes_play, average='macro')
        
        self.val_acc_vid = torchmetrics.Accuracy(**metrics_params)
        self.val_f1_vid = torchmetrics.F1Score(**metrics_params, average='macro')
        self.val_f1_play = torchmetrics.F1Score(task="multiclass", num_classes=num_classes_play, average='macro')

    def forward(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x_l3 = self.stages[2](x) 
        x_l4 = self.stages[3](x_l3)
        
        stat_l3 = self.pool_l3(x_l3) 
        stat_l4 = self.pool_l4(x_l4)
        
        combined = torch.cat([stat_l3, stat_l4], dim=1)
        feat = self.shared_fc(combined)
        
        return self.head_vid(feat), self.head_play(feat), feat

    def training_step(self, batch, batch_idx):
        x, y_vid, y_play = batch
        logits_vid, logits_play, _ = self(x) 
        
        loss_vid = self.criterion_vid(logits_vid, y_vid)
        loss_play = self.criterion_play(logits_play, y_play)
        loss = loss_vid + self.factor * loss_play
        
        preds_vid = torch.argmax(logits_vid, dim=1)
        preds_play = torch.argmax(logits_play, dim=1)
        
        # 更新指标
        self.train_acc_vid(preds_vid, y_vid)
        self.train_f1_vid(preds_vid, y_vid)
        self.train_f1_play(preds_play, y_play)
        
        # Log with sync_dist=True for Multi-GPU
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/acc_vid', self.train_acc_vid, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/f1_vid', self.train_f1_vid, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_vid, y_play = batch
        logits_vid, logits_play, _ = self(x)
        
        loss_vid = self.criterion_vid(logits_vid, y_vid)
        loss_play = self.criterion_play(logits_play, y_play)
        loss = loss_vid + self.factor * loss_play
        
        preds_vid = torch.argmax(logits_vid, dim=1)
        preds_play = torch.argmax(logits_play, dim=1)
        
        # Video Metrics
        self.val_acc_vid(preds_vid, y_vid)
        self.val_f1_vid(preds_vid, y_vid)
        
        # Playlist Metrics
        self.val_f1_play(preds_play, y_play)
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_vid', self.val_f1_vid, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc_vid', self.val_acc_vid, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/f1_play', self.val_f1_play, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=self.lr, 
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25.0
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

class AcademicEvaluator:
    """
    SALT Evaluation Suite: 
    Designed for SALT (Size And Logarithmic Timing) method.
    Includes AP@K, Recall@K, AUC, ECE, t-SNE, Semantic Sorted CM, etc.
    """
    def __init__(self, model, dataloader, output_path, device=None):
        self.model = model
        self.dataloader = dataloader
        self.output_path = output_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_path, exist_ok=True)
        
        # Store results
        self.results = {
            'v_true': [], 'v_pred': [], 'v_probs': [], 'features': [],
            'p_true': [], 'p_pred': [], 'p_probs': []
        }
        
        # Plot styling
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    def run_inference(self):
        """Execute inference, collect logits and features"""
        self.model.eval().to(self.device)
        print("[SALT Evaluator] Running Inference on Test Set...")
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Inference"):
                x, y_v, y_p = [t.to(self.device) for t in batch]
                
                # Model forward pass
                l_v, l_p, feats = self.model(x)
                
                # Collect Video Results
                self.results['v_true'].extend(y_v.cpu().numpy())
                self.results['v_pred'].extend(l_v.argmax(1).cpu().numpy())
                self.results['v_probs'].extend(F.softmax(l_v, dim=1).cpu().numpy())
                self.results['features'].extend(feats.cpu().numpy())
                
                # Collect Playlist Results
                self.results['p_true'].extend(y_p.cpu().numpy())
                self.results['p_pred'].extend(l_p.argmax(1).cpu().numpy())
                self.results['p_probs'].extend(F.softmax(l_p, dim=1).cpu().numpy())

        # Convert to numpy arrays
        for k in self.results: 
            self.results[k] = np.array(self.results[k])

    def _compute_ece(self, probs, labels, n_bins=15):
        """Compute Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        
        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def _compute_ap_at_k(self, y_true, y_probs, k_list=[1, 5, 10]):
        """
        Compute Average Precision @ K.
        """
        sorted_indices = np.argsort(y_probs, axis=1)[:, ::-1]
        metrics = {}
        
        for k in k_list:
            ap_sum = 0.0
            for i in range(len(y_true)):
                top_k = sorted_indices[i, :k]
                if y_true[i] in top_k:
                    rank = np.where(top_k == y_true[i])[0][0]
                    ap_sum += 1.0 / (rank + 1)
                else:
                    ap_sum += 0.0
            metrics[f'AP@{k}'] = ap_sum / len(y_true)
        return metrics

    def compute_metrics(self):
        """Compute detailed metrics for both Video and Playlist tasks"""
        print("Calculating SALT Performance Metrics...")
        metrics = {}
        
        # ==========================================
        # 1. Video Metrics (Main Task)
        # ==========================================
        v_true = self.results['v_true']
        v_pred = self.results['v_pred']
        v_probs = self.results['v_probs']
        num_classes_v = v_probs.shape[1]

        unmonitored_label_idx = np.max(v_true) 
        
        # Identify Monitored Samples
        monitored_mask = v_true != unmonitored_label_idx
        
        if np.sum(monitored_mask) > 0:
            acc_mon_only = accuracy_score(v_true[monitored_mask], v_pred[monitored_mask])
            print(f"  [Insight] Acc on Monitored Samples Only: {acc_mon_only:.4f}")
            metrics['v_acc_monitored_only'] = acc_mon_only

        # Identify Unmonitored Samples
        unmonitored_mask = ~monitored_mask
        
        if np.sum(unmonitored_mask) > 0:
            acc_unmon_only = accuracy_score(v_true[unmonitored_mask], v_pred[unmonitored_mask])
            print(f"  [Insight] Acc on Unmonitored Samples (Rejection Rate): {acc_unmon_only:.4f}")
            metrics['v_acc_unmonitored_only'] = acc_unmon_only
        
        metrics['v_acc'] = accuracy_score(v_true, v_pred)
        metrics['v_f1_macro'] = f1_score(v_true, v_pred, average='macro')
        metrics['v_f1_weighted'] = f1_score(v_true, v_pred, average='weighted')
        metrics['v_kappa'] = cohen_kappa_score(v_true, v_pred)
        metrics['v_ece'] = self._compute_ece(v_probs, v_true)
        
        # --- Video AP@K ---
        v_ap_metrics = self._compute_ap_at_k(v_true, v_probs, k_list=[1, 5, 10])
        for k, v in v_ap_metrics.items():
            metrics[f'v_{k}'] = v

        # --- Video Recall@K---
        sorted_indices = np.argsort(v_probs, axis=1)[:, ::-1]
        k_values = [1, 5, 10, 20]
        for k in k_values:
            if k > num_classes_v: continue
            hits = [1 if v_true[i] in sorted_indices[i, :k] else 0 for i in range(len(v_true))]
            metrics[f'v_recall@{k}'] = np.mean(hits)

        # --- Video AUC (OvR) ---
        try:
            v_true_bin = label_binarize(v_true, classes=range(num_classes_v))
            if v_true_bin.shape[1] == num_classes_v:
                metrics['v_auc_ovr'] = roc_auc_score(v_true_bin, v_probs, multi_class='ovr', average='macro')
        except Exception as e: 
            print(f" AUC calculation skipped: {e}")

        # ==========================================
        # 2. Playlist Metrics (Auxiliary Task)
        # ==========================================
        p_true = self.results['p_true']
        p_pred = self.results['p_pred']
        p_probs = self.results['p_probs']
        
        metrics['p_acc'] = accuracy_score(p_true, p_pred)
        metrics['p_f1_macro'] = f1_score(p_true, p_pred, average='macro')
        metrics['p_f1_weighted'] = f1_score(p_true, p_pred, average='weighted')
        metrics['p_kappa'] = cohen_kappa_score(p_true, p_pred)
        
        # Playlist AP@K
        p_ap_metrics = self._compute_ap_at_k(p_true, p_probs, k_list=[1]) 
        metrics['p_AP@1'] = p_ap_metrics['AP@1']

        # ==========================================
        # 3. Print & Save
        # ==========================================
        print(f"\n [Video Task] Performance:")
        print(f"    Accuracy (Recall@1): {metrics['v_acc']:.4f}")
        print(f"    F1 Macro: {metrics['v_f1_macro']:.4f}")
        print(f"    Average Precision (AP):")
        print(f"      AP@1: {metrics['v_AP@1']:.4f} | AP@5: {metrics['v_AP@5']:.4f} | AP@10: {metrics['v_AP@10']:.4f}")
        print(f"    Recall (Top-K):")
        print(f"      R@5: {metrics.get('v_recall@5',0):.4f} | R@10: {metrics.get('v_recall@10',0):.4f} | R@20: {metrics.get('v_recall@20',0):.4f}")
        print(f"    ECE: {metrics['v_ece']:.4f} (Lower is better)")
        if 'v_auc_ovr' in metrics:
            print(f"    AUC (OvR): {metrics['v_auc_ovr']:.4f}")

        print(f"\n [Playlist Task] Performance:")
        print(f"    Accuracy: {metrics['p_acc']:.4f}")
        print(f"    F1 Macro: {metrics['p_f1_macro']:.4f}")
        print(f"    Kappa: {metrics['p_kappa']:.4f}")

        # Save metrics
        with open(os.path.join(self.output_path, 'salt_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save CSV Report
        report = classification_report(v_true, v_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.output_path, 'video_classification_report.csv'))
        
        return metrics

    def plot_visualizations(self):
        """Generate visualizations"""
        print(" Generating Visualizations...")
        
        # 1. Playlist Confusion Matrix
        self._plot_confusion_matrix(
            self.results['p_true'], self.results['p_pred'], 
            title="SALT Playlist Confusion Matrix", 
            filename="cm_playlist.png", 
            figsize=(8, 6), cmap='Blues'
        )

        # 2. Video Confusion Matrix (Semantic Sorted, Blues)
        self._plot_sorted_video_cm()

        # 3. t-SNE (Colored by Playlist)
        self._plot_semantic_tsne()

        # 4. Reliability Diagram
        self._plot_reliability_diagram()
        
        # 5. Top-K Confusion Pairs
        self._analyze_top_confusions()

    def _plot_confusion_matrix(self, y_true, y_pred, title, filename, figsize, labels=None, cmap='Blues'):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm_norm, cmap=cmap, annot=False if cm.shape[0] > 20 else True, fmt='.2f', 
                    cbar_kws={'label': 'Normalized Accuracy'})
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename), dpi=300)
        plt.close()

    def _plot_sorted_video_cm(self):
        v_true = self.results['v_true']
        v_pred = self.results['v_pred']
        p_true = self.results['p_true'] 

        unique_videos = np.unique(v_true)
        # Mapping: Video -> Playlist
        vid_to_pid = {}
        for v, p in zip(v_true, p_true):
            vid_to_pid[v] = p
            
        # Sort videos by Playlist ID, then by Video ID
        sorted_vids = sorted(unique_videos, key=lambda x: (vid_to_pid.get(x, -1), x))
        remap = {old_id: new_id for new_id, old_id in enumerate(sorted_vids)}
        
        v_true_mapped = [remap[x] for x in v_true if x in remap]
        v_pred_mapped = [remap[x] for x in v_pred if x in remap]
        
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(v_true_mapped, v_pred_mapped, labels=range(len(sorted_vids)))
        
        from matplotlib.colors import LogNorm
        plt.imshow(cm + 1, norm=LogNorm(vmin=1, vmax=cm.max()), cmap='Blues')
        
        plt.title(f"SALT Video Confusion Matrix (Sorted by Playlist)\n{len(sorted_vids)} Classes")
        plt.colorbar(label='Log Scale Count')
        plt.xlabel('Predicted Video ID (Sorted)')
        plt.ylabel('True Video ID (Sorted)')
        
        # Draw boundaries for Playlists
        current_pid = -1
        boundary_indices = []
        for idx, vid in enumerate(sorted_vids):
            pid = vid_to_pid.get(vid, -1)
            if pid != current_pid:
                boundary_indices.append(idx)
                current_pid = pid
        
        for b in boundary_indices[1:]:
            plt.axhline(y=b, color='black', linewidth=0.5, alpha=0.3)
            plt.axvline(x=b, color='black', linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'cm_video_semantic_sorted.png'), dpi=300)
        plt.close()

    def _plot_semantic_tsne(self):
        # Sample points
        n_samples = min(5000, len(self.results['features']))
        if n_samples == 0: return
        indices = np.random.choice(len(self.results['features']), n_samples, replace=False)
        
        feats = self.results['features'][indices]
        p_labels = self.results['p_true'][indices]
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        feats_2d = tsne.fit_transform(feats)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=p_labels, cmap='tab10', alpha=0.7, s=20)
        plt.legend(*scatter.legend_elements(), title="Playlist ID", loc="best")
        plt.title("t-SNE of SALT Features (Colored by Playlist)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'tsne_semantic.png'), dpi=300)
        plt.close()

    def _plot_reliability_diagram(self):
        probs = self.results['v_probs']
        preds = np.argmax(probs, axis=1)
        labels = self.results['v_true']
        
        confidences = np.max(probs, axis=1)
        accuracies = (preds == labels).astype(int)
        
        prob_true, prob_pred = calibration_curve(accuracies, confidences, n_bins=10, strategy='uniform')
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='.', label='SALT Model')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Fraction of Positives (Accuracy)')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'reliability_diagram.png'), dpi=300)
        plt.close()

    def _analyze_top_confusions(self):
        y_true = self.results['v_true']
        y_pred = self.results['v_pred']
        
        errors = y_true != y_pred
        error_pairs = list(zip(y_true[errors], y_pred[errors]))
        
        if not error_pairs:
            return

        from collections import Counter
        pair_counts = Counter(error_pairs)
        top_10 = pair_counts.most_common(10)
        
        pairs = [f"{t}->{p}" for (t, p), c in top_10]
        counts = [c for (t, p), c in top_10]
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=counts, y=pairs, palette='viridis')
        plt.title('Top 10 Confused Video Pairs (True -> Pred)')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'top_confusions.png'), dpi=300)
        plt.close()

    def compute_roc_and_plot(self):
        """
        Calculate and plot Multi-class ROC (One-vs-Rest)
        Saves raw ROC data for combined plotting later.
        """
        print("Calculating ROC & FPR metrics...")
        
        y_true = self.results['v_true']
        y_probs = self.results['v_probs']
        n_classes = y_probs.shape[1]
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # --- Micro-average ROC ---
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # --- Macro-average ROC ---
        all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            f, t, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            mean_tpr += np.interp(all_fpr, f, t)
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Save ROC data
        self._save_roc_data(fpr, tpr, roc_auc)

        # Metrics at low FPR
        fpr_targets = [0.001, 0.01, 0.05] 
        print("\n Security Metrics (Performance at low False Positive Rates):")
        for ft in fpr_targets:
            idx = np.argmin(np.abs(fpr["micro"] - ft))
            tpr_at_fpr = tpr["micro"][idx]
            print(f"   🔹 TPR @ FPR={ft*100}%: {tpr_at_fpr*100:.2f}% (Micro-avg)")

        self._plot_roc_curve(fpr, tpr, roc_auc)

    def _save_roc_data(self, fpr, tpr, roc_auc):
        save_path = os.path.join(self.output_path, 'roc_data.npz')
        np.savez(
            save_path,
            fpr_micro=fpr["micro"],
            tpr_micro=tpr["micro"],
            auc_micro=roc_auc["micro"],
            fpr_macro=fpr["macro"],
            tpr_macro=tpr["macro"],
            auc_macro=roc_auc["macro"]
        )
        print(f" ROC Raw Data saved to {save_path}")

    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
                 color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Recall)', fontsize=14)
        plt.title('Multi-class ROC Curve (One-vs-Rest)', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Zoom-in for Low FPR
        ax_ins = plt.axes([0.6, 0.2, 0.25, 0.25]) 
        ax_ins.plot(fpr["micro"], tpr["micro"], color='deeppink', linewidth=2)
        ax_ins.set_xlim([0, 0.1]) 
        ax_ins.set_ylim([0.8, 1.0])
        ax_ins.set_title('Zoom: Low FPR', fontsize=10)
        ax_ins.grid(True, linestyle=':', alpha=0.5)
        
        save_path = os.path.join(self.output_path, 'roc_curve_ccs_style.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC Curve saved to {save_path}")

    def run(self):
        """Execute full evaluation pipeline"""
        self.run_inference()
        self.compute_metrics()
        self.compute_roc_and_plot()
        self.plot_visualizations()
        print(f"Evaluation Done! Results saved to {self.output_path}")

# ==============================================================================
# 3. Helpers
# ==============================================================================
def get_combined_samples_per_play(monitored_train_prefix, unmonitored_train_prefix, is_open_world):
    """Calculate combined class distribution for Loss weighting"""
    p_mon = f"{monitored_train_prefix}_label_playlist_code.npy"
    if os.path.exists(p_mon):
        counts_mon = np.bincount(np.load(p_mon, mmap_mode='r').astype(int))
    else:
        counts_mon = np.array([1]*7) 

    if len(counts_mon) < 8:
        counts_mon = np.pad(counts_mon, (0, 7 - len(counts_mon)), 'constant')

    if not is_open_world:
        return np.where(counts_mon == 0, 1, counts_mon)

    p_unmon = f"{unmonitored_train_prefix}_label_playlist_code.npy"
    if os.path.exists(p_unmon):
        labels_un = np.load(p_unmon, mmap_mode='r').astype(int)
        counts_un = np.bincount(labels_un)
        
        max_len = max(len(counts_mon), len(counts_un))
        combined = np.zeros(max_len, dtype=int)
        combined[:len(counts_mon)] += counts_mon
        combined[:len(counts_un)] += counts_un
        
        combined = np.where(combined == 0, 1, combined)
        print(f"[Class Balance] Mixed Distribution: {combined}")
        return combined
    else:
        print(f"Warning: Open World enabled but file not found: {p_unmon}")
        return np.where(counts_mon == 0, 1, counts_mon)
    
def get_samples_per_play(path):
    p = f"{path}_label_playlist_code.npy"
    if not os.path.exists(p): return [1]*7
    counts = np.bincount(np.load(p, mmap_mode='r').astype(int))
    if len(counts) < 7: counts = np.pad(counts, (0, 7-len(counts)), 'constant', constant_values=1)
    return np.where(counts==0, 1, counts)

def main():
    parser = argparse.ArgumentParser(description="Multi-Task TrafficNet Evaluator")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help="Train or Eval")
    parser.add_argument('--rep', type=str, default='SALT', help="Representation (Default: SALT)")
    parser.add_argument('--proto', type=str, default='tor', help="Protocol")
    parser.add_argument('--plat', type=str, default='youtube', help="Platform")
    parser.add_argument('--checkpoint', type=str, help="Path to .ckpt file (Required for eval)")
    parser.add_argument('--open_world', default=False, action='store_true', help="Enable Open World (N+1 Class) Mode")
    args = parser.parse_args()

    # Path Config
    mod_rep = args.rep
    
    mon_dir = os.path.join(BASE_PATH, f"{mod_rep}/monitored_{args.proto}")
    unmon_dir = os.path.join(BASE_PATH, f"{mod_rep}/unmonitored_{args.proto}")
    
    key = f"{args.rep}_{args.proto}_{args.plat}"
    input_length = SHAPES.get(args.rep, 25000)
    in_channels = 2 if args.rep == 'SALT' or 'dschuster' in args.rep else 1
    
    # Class Definitions
    if args.open_world:
        num_classes_vid = 290
        num_classes_play = 8
        task_suffix = "_OpenWorld"
    else:
        # Closed World
        num_classes_vid = 289
        num_classes_play = 7
        task_suffix = ""

    train_prefix_mon = os.path.join(mon_dir, 'train')
    train_prefix_unmon = os.path.join(unmon_dir, 'train')
    samples_per_play = get_combined_samples_per_play(train_prefix_mon, train_prefix_unmon, args.open_world)

    # ==========================================
    # 2. Train Mode
    # ==========================================
    if args.mode == 'train':
        hype = BEST_HYPERPARAMETERS.get(key, BEST_HYPERPARAMETERS['SALT_tor_youtube'])
        print(f"Training {key} {task_suffix} with params: {hype}")
        
        # Load Monitored Data
        train_ds = MemmapDataset(os.path.join(mon_dir, 'train'), args.rep, input_length, 'vid_adjusted', augment=False)
        val_ds = MemmapDataset(os.path.join(mon_dir, 'val'), args.rep, input_length, 'vid_adjusted')
        
        # Load Unmonitored Data if Open World
        if args.open_world:
            print("[Dataset] Mixing Unmonitored data into Training set...")
            try:
                train_ds_un = MemmapDataset(os.path.join(unmon_dir, 'train'), args.rep, input_length, 'vid_adjusted', augment=False)
                val_ds_un = MemmapDataset(os.path.join(unmon_dir, 'val'), args.rep, input_length, 'vid_adjusted')
                
                train_ds = ConcatDataset([train_ds, train_ds_un])
                val_ds = ConcatDataset([val_ds, val_ds_un])
            except FileNotFoundError as e:
                print(f"Failed to load Unmonitored data: {e}")
                return

        # Dataloader
        train_loader = DataLoader(train_ds, batch_size=hype['batch_size'], shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=hype['batch_size'], num_workers=12)
        
        # Model Init
        model = MultiTaskTrafficNet(
            in_channels, 
            input_length, 
            num_classes_vid, 
            num_classes_play,
            samples_per_play, 
            hype, 
            lr=hype['lr']
        )
        
        # Callbacks
        ckpt_cb = ModelCheckpoint(
            monitor='val/f1_vid', 
            mode='max', 
            filename='best-{epoch}-{val/f1_vid:.4f}', 
            save_top_k=1
        )
        early_stop = EarlyStopping('val/f1_vid', patience=15, mode='max')
        
        trainer = pl.Trainer(
            max_epochs=80, 
            accelerator="gpu", 
            devices=[0], 
            callbacks=[ckpt_cb, early_stop],
            log_every_n_steps=10
        )
        
        trainer.fit(model, train_loader, val_loader)
        print(f"Training finished. Best model path: {ckpt_cb.best_model_path}")
        
        eval_checkpoint = ckpt_cb.best_model_path
        model = MultiTaskTrafficNet.load_from_checkpoint(eval_checkpoint, samples_per_class_play=samples_per_play)

    # ==========================================
    # 3. Eval Mode
    # ==========================================
    else:
        if not args.checkpoint:
            print(" Eval mode requires --checkpoint path"); return
        eval_checkpoint = args.checkpoint
        print(f"Loading checkpoint: {eval_checkpoint}")
        model = MultiTaskTrafficNet.load_from_checkpoint(eval_checkpoint, samples_per_class_play=samples_per_play)
    test_ds = MemmapDataset(os.path.join(mon_dir, 'test'), args.rep, input_length, 'vid_adjusted', augment=False)
    
    if args.open_world:
        print("[Dataset] Mixing Unmonitored data into Test set...")
        try:
            test_ds_un = MemmapDataset(os.path.join(unmon_dir, 'test'), args.rep, input_length, 'vid_adjusted', augment=False)
            test_ds = ConcatDataset([test_ds, test_ds_un])
        except FileNotFoundError:
            print("Unmonitored test set not found, evaluating on Monitored set only.")

    test_loader = DataLoader(test_ds, batch_size=128, num_workers=16, shuffle=False)
    
    task_name = f"{key}_{args.mode}_results{task_suffix}"
    evaluator = AcademicEvaluator(model, test_loader, os.path.join(OUTPUT_DIR, task_name))
    
    evaluator.run()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
    
    # Example Commands for Reviewers:
    # python evaluate.py --mode train --rep SALT --proto tor --plat youtube
    # python evaluate.py --mode eval --rep SALT --proto tor --plat youtube --checkpoint ./checkpoints/best_model.ckpt