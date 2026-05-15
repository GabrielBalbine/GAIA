"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██████╗  █████╗ ██╗ █████╗     ██╗     ███████╗████████╗███╗   ███╗      ║
║    ██╔════╝ ██╔══██╗██║██╔══██╗    ██║     ██╔════╝╚══██╔══╝████╗ ████║      ║
║    ██║  ███╗███████║██║███████║    ██║     ███████╗   ██║   ██╔████╔██║      ║
║    ██║   ██║██╔══██║██║██╔══██║    ██║     ╚════██║   ██║   ██║╚██╔╝██║      ║
║    ╚██████╔╝██║  ██║██║██║  ██║    ███████╗███████║   ██║   ██║ ╚═╝ ██║      ║
║     ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝    ╚══════╝╚══════╝   ╚═╝   ╚═╝     ╚═╝      ║
║                                                                               ║
║         ★★★ GAIA-LSTM — Temporal Behavioral Pattern Classifier ★★★          ║
║                                                                               ║
║  Classificador temporal de padrões comportamentais para análise de            ║
║  interação cuidador-criança como ferramenta auxiliar de prognóstico TEA.      ║
║                                                                               ║
║  INPUT:  JSONs frame-a-frame gerados pelo TITAN v58 pipeline                 ║
║  OUTPUT: Relatório de prognóstico com métricas por dimensão + P(TEA)         ║
║                                                                               ║
║  ARQUITETURA:                                                                ║
║   • Feature Extractor: 14 features/frame em 4 dimensões comportamentais     ║
║   • Bi-LSTM (2 camadas, 128 unidades) + Temporal Attention Pooling          ║
║   • Sliding windows de ~10s (300 frames) com stride 50%                     ║
║   • Validação: Leave-One-Subject-Out (LOSO) cross-validation                ║
║   • Saída: P(TEA) por janela → agregado por sessão + métricas descritivas   ║
║                                                                               ║
║  NOTA ÉTICA: Esta é uma ferramenta AUXILIAR de prognóstico.                  ║
║  A avaliação final deve ser realizada por profissionais qualificados.        ║
║                                                                               ║
║  v1.0 — 2026                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import glob
import time
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore', category=UserWarning)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 0: CONFIGURAÇÃO                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class LSTMConfig:
    # ── Caminhos (relativos ao script) ──
    output_base: str = "output"
    lstm_output_dir: str = "lstm_output"
    groups: List[str] = field(default_factory=lambda: ["Neurotipico", "TEA"])
    group_labels: Dict[str, int] = field(default_factory=lambda: {"Neurotipico": 0, "TEA": 1})

    # ── Features ──
    num_features: int = 14
    feature_names: List[str] = field(default_factory=lambda: [
        "g_looking_score",        # 0  — Guardian gaze score
        "c_looking_score",        # 1  — Child gaze score
        "mutual_attention",       # 2  — Both looking (0/1 → float)
        "gaze_asymmetry",         # 3  — g_score - c_score
        "interpersonal_distance", # 4  — Euclidean head-to-head (normalized)
        "distance_velocity",      # 5  — Δdistance/Δframe
        "g_shoulder_tilt",        # 6  — Guardian shoulder tilt (deg, normalized)
        "c_shoulder_tilt",        # 7  — Child shoulder tilt
        "c_hand_distance",        # 8  — Child wrist spread (normalized)
        "c_hand_distance_vel",    # 9  — Δhand_dist/Δframe (motor repetitiveness)
        "g_attn_episode_len",     # 10 — Running count of G looking streak
        "c_attn_episode_len",     # 11 — Running count of C looking streak
        "g_head_movement",        # 12 — Guardian head position delta (restlessness)
        "c_head_movement",        # 13 — Child head position delta
    ])

    # ── Windowing ──
    window_size: int = 300          # ~10s @ 30fps
    window_stride: int = 150        # 50% overlap
    min_valid_ratio: float = 0.65   # skip windows with >35% missing frames

    # ── Filtro de qualidade por sessão ──
    min_valid_pct_nt: float = 50.0   # mínimo % frames válidos para NT
    min_valid_pct_tea: float = 40.0  # mínimo % frames válidos para TEA (limiar relaxado)

    # ── Modelo ──
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

    # ── Treinamento ──
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 150
    patience: int = 20              # early stopping
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8

    # ── Reprodutibilidade ──
    seed: int = 42

    # ── Normalização ──
    normalize_per_session: bool = True  # z-score per session

    def __post_init__(self):
        assert len(self.feature_names) == self.num_features


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
        print(f"  Device: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        dev = torch.device('cpu')
        print("  Device: CPU")
    return dev


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 0.5: HELPER — EXTRAÇÃO DE ID DO PARTICIPANTE                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def extract_participant_id(session_name: str) -> str:
    """
    Extrai o ID do participante a partir do nome da sessão.

    Regras:
      - Começa com 'H' → primeiros 4 caracteres (H016, H017, H018, H019, H024)
      - Começa com 'C' → pega até o primeiro underscore ou espaço (C1009, C1012, etc.)

    Exemplos:
      'H017_Video Cam 1_2 week'       → 'H017'
      'C1009 PDI Coach 1 - cam1'      → 'C1009'
      'C1009_PDI_Coach_1_cam1'        → 'C1009'
      'H016 [2 week] Cam 1'           → 'H016'
    """
    name = session_name.strip()
    if name.startswith('H'):
        return name[:4]
    elif name.startswith('C'):
        # Pega tudo até o primeiro espaço ou underscore
        match = re.match(r'^(C\d+)', name)
        if match:
            return match.group(1)
        # Fallback: até primeiro separador
        for i, ch in enumerate(name):
            if ch in (' ', '_') and i > 0:
                return name[:i]
        return name  # sessão inteira se não achar separador
    else:
        # Fallback genérico: até primeiro separador
        match = re.match(r'^([A-Za-z0-9]+)', name)
        return match.group(1) if match else name


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 1: FEATURE EXTRACTOR — JSON → raw feature matrix                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _get_person(frame: dict, role: str) -> Optional[dict]:
    """Extrai dados de uma pessoa pelo role."""
    for p in frame.get('people', []):
        if p.get('role') == role:
            return p
    return None


def extract_features_from_json(json_path: str, config: LSTMConfig) -> Optional[Dict]:
    """
    Lê um JSON do TITAN e extrai a matriz de features frame-a-frame.

    Returns:
        dict com 'features' (np.ndarray shape [N, 14]),
                  'timestamps' (np.ndarray shape [N]),
                  'raw_frames' (list of dicts para métricas descritivas),
                  'session_name' (str),
                  'fps_estimate' (float)
        ou None se falhar.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            frames = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Could not load {json_path}: {e}")
        return None

    if not frames or len(frames) < config.window_size // 2:
        print(f"  [WARN] {json_path}: too few frames ({len(frames)}), skipping.")
        return None

    session_name = os.path.splitext(os.path.basename(json_path))[0]
    if session_name.startswith("json_"):
        session_name = session_name[5:]

    # ── Estimar FPS a partir dos timestamps ──
    timestamps = [fr['timestamp_s'] for fr in frames]
    frame_ids = [fr['frame_id'] for fr in frames]
    if len(timestamps) >= 2:
        dt = timestamps[-1] - timestamps[0]
        df = frame_ids[-1] - frame_ids[0]
        fps_estimate = df / dt if dt > 0 else 30.0
    else:
        fps_estimate = 30.0

    N = len(frames)
    features = np.full((N, config.num_features), np.nan, dtype=np.float32)

    # ── Estado para features temporais ──
    g_streak = 0
    c_streak = 0
    prev_g_pos = None
    prev_c_pos = None
    prev_dist = None
    prev_c_hand = None

    for i, fr in enumerate(frames):
        guardian = _get_person(fr, 'GUARDIAN')
        child = _get_person(fr, 'CHILD')

        if guardian is None or child is None:
            continue

        g_score = float(guardian.get('looking_score', 0.0))
        c_score = float(child.get('looking_score', 0.0))
        g_looking = bool(guardian.get('is_looking', False))
        c_looking = bool(child.get('is_looking', False))
        mutual = 1.0 if fr.get('mutual_attention', False) else 0.0

        # Posições da cabeça
        g_pos = np.array(guardian.get('pos_head', [0, 0]), dtype=np.float32)
        c_pos = np.array(child.get('pos_head', [0, 0]), dtype=np.float32)

        # Distância interpessoal
        dist = float(np.linalg.norm(g_pos - c_pos))

        # Velocidade da distância
        dist_vel = 0.0
        if prev_dist is not None:
            dist_vel = dist - prev_dist

        # Postura
        g_posture = guardian.get('posture', {})
        c_posture = child.get('posture', {})

        g_shoulder = float(g_posture.get('shoulder_tilt_degrees', 0.0))
        c_shoulder = float(c_posture.get('shoulder_tilt_degrees', 0.0))
        c_hand = float(c_posture.get('hand_distance_px', 0.0))

        # Velocidade mãos da criança
        c_hand_vel = 0.0
        if prev_c_hand is not None and c_hand > 0 and prev_c_hand > 0:
            c_hand_vel = c_hand - prev_c_hand

        # Streaks de atenção
        g_streak = (g_streak + 1) if g_looking else 0
        c_streak = (c_streak + 1) if c_looking else 0

        # Movimento da cabeça (delta de posição)
        g_head_mov = 0.0
        c_head_mov = 0.0
        if prev_g_pos is not None:
            g_head_mov = float(np.linalg.norm(g_pos - prev_g_pos))
        if prev_c_pos is not None:
            c_head_mov = float(np.linalg.norm(c_pos - prev_c_pos))

        # ── Montar vetor de features ──
        features[i] = [
            g_score,                           # 0
            c_score,                           # 1
            mutual,                            # 2
            g_score - c_score,                 # 3  gaze_asymmetry
            dist,                              # 4
            dist_vel,                          # 5
            g_shoulder,                        # 6
            c_shoulder,                        # 7
            c_hand,                            # 8
            c_hand_vel,                        # 9
            float(g_streak),                   # 10
            float(c_streak),                   # 11
            g_head_mov,                        # 12
            c_head_mov,                        # 13
        ]

        prev_g_pos = g_pos.copy()
        prev_c_pos = c_pos.copy()
        prev_dist = dist
        prev_c_hand = c_hand if c_hand > 0 else prev_c_hand

    # ── Verificar quantidade de frames válidos ──
    valid_mask = ~np.isnan(features[:, 0])
    n_valid = valid_mask.sum()
    if n_valid < config.window_size // 2:
        print(f"  [WARN] {session_name}: only {n_valid}/{N} valid frames, skipping.")
        return None

    # ── Interpolar NaNs (forward fill + backward fill + zero) ──
    for col in range(config.num_features):
        arr = features[:, col]
        nans = np.isnan(arr)
        if nans.all():
            features[:, col] = 0.0
            continue
        if nans.any():
            # Forward fill
            idx = np.where(~nans, np.arange(N), 0)
            np.maximum.accumulate(idx, out=idx)
            features[:, col] = arr[idx]
            # Backward fill remaining leading NaNs
            nans2 = np.isnan(features[:, col])
            if nans2.any():
                first_valid = np.argmax(~nans)
                features[:nans2.sum(), col] = arr[first_valid]

    # ── Total real de frames do vídeo (pelo frame_id, não pelo len do JSON) ──
    # O TITAN só escreve no JSON os frames com detecção válida,
    # então len(frames) ≠ total de frames do vídeo.
    # O frame_id preserva o índice original do vídeo.
    n_total_video = frame_ids[-1] + 1 if frame_ids else N

    return {
        'features': features,
        'timestamps': np.array(timestamps, dtype=np.float32),
        'raw_frames': frames,
        'session_name': session_name,
        'fps_estimate': fps_estimate,
        'n_valid': int(n_valid),
        'n_total': n_total_video,
        'n_json_frames': N,  # quantos frames o JSON realmente tem
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 2: NORMALIZAÇÃO + WINDOWING                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def normalize_session(features: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalização por sessão.
    Returns: (normalized_features, means, stds)
    """
    means = np.nanmean(features, axis=0)
    stds = np.nanstd(features, axis=0)
    stds = np.where(stds < eps, 1.0, stds)  # evita divisão por zero
    normalized = (features - means) / stds
    return normalized, means, stds


def create_windows(features: np.ndarray, config: LSTMConfig) -> List[np.ndarray]:
    """
    Sliding window sobre a matriz de features.
    Retorna lista de arrays shape (window_size, num_features).
    Descarta janelas com muitos NaNs.
    """
    N = features.shape[0]
    windows = []

    for start in range(0, N - config.window_size + 1, config.window_stride):
        window = features[start:start + config.window_size]

        # Verificar validade (NaNs remanescentes pós-interpolação = gaps reais)
        valid_ratio = np.mean(~np.isnan(window[:, 0]))
        if valid_ratio < config.min_valid_ratio:
            continue

        # Substituir NaNs residuais por 0
        window = np.nan_to_num(window, nan=0.0)
        windows.append(window)

    return windows


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 3: DATASET                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class GAIADataset(Dataset):
    """
    Dataset de janelas temporais para classificação NT vs TEA.
    Cada sample = (window_tensor [W, F], label, session_id).
    """

    def __init__(self):
        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.session_ids: List[str] = []
        self.session_meta: Dict[str, Dict] = {}  # session_name → metadados

    def add_session(self, session_name: str, windows: List[np.ndarray],
                    label: int, meta: Dict = None):
        for w in windows:
            self.windows.append(w)
            self.labels.append(label)
            self.session_ids.append(session_name)
        if meta:
            self.session_meta[session_name] = meta

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.windows[idx])
        y = torch.FloatTensor([self.labels[idx]])
        return x, y, self.session_ids[idx]

    def get_session_names(self) -> List[str]:
        return sorted(set(self.session_ids))

    def get_indices_for_sessions(self, sessions: List[str]) -> List[int]:
        return [i for i, s in enumerate(self.session_ids) if s in sessions]

    def get_class_weights(self, indices: List[int]) -> torch.Tensor:
        """Peso para balanceamento de classes nos índices dados."""
        labels = [self.labels[i] for i in indices]
        n0 = sum(1 for l in labels if l == 0)
        n1 = sum(1 for l in labels if l == 1)
        if n0 == 0 or n1 == 0:
            return torch.tensor([1.0])
        return torch.tensor([n0 / n1], dtype=torch.float32)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 4: MODELO — Bi-LSTM + Temporal Attention                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TemporalAttention(nn.Module):
    """
    Mecanismo de atenção temporal: aprende quais timesteps
    dentro da janela são mais discriminativos.

    Retorna: context vector (batch, hidden_dim) e weights (batch, seq_len).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        lstm_output: (batch, seq_len, hidden_dim)
        Returns: (context, attention_weights)
        """
        energy = self.v(torch.tanh(self.W(lstm_output)))  # (batch, seq, 1)
        weights = F.softmax(energy, dim=1)                 # (batch, seq, 1)
        context = (weights * lstm_output).sum(dim=1)       # (batch, hidden)
        return context, weights.squeeze(-1)                # weights: (batch, seq)


class GAIAClassifier(nn.Module):
    """
    Bi-LSTM + Temporal Attention → P(TEA).

    Input:  (batch, window_size, num_features)
    Output: (batch, 1) — probabilidade de TEA
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.input_bn = nn.BatchNorm1d(config.num_features)

        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        self.attention = TemporalAttention(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x: (batch, seq_len, features)
        Returns: logits (batch, 1)  [apply sigmoid externally]
                 optionally: attention_weights (batch, seq_len)
        """
        # BatchNorm sobre features (transpõe para [B, F, T] e volta)
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.input_bn(x)
        x = x.transpose(1, 2)          # (B, T, F)

        lstm_out, _ = self.lstm(x)      # (B, T, H*2)

        context, attn_weights = self.attention(lstm_out)  # (B, H*2), (B, T)

        logits = self.classifier(context)  # (B, 1)

        if return_attention:
            return logits, attn_weights
        return logits


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 5: TREINAMENTO — LOSO Cross-Validation                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def train_one_fold(
    model: GAIAClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LSTMConfig,
    device: torch.device,
    pos_weight: torch.Tensor,
    fold_name: str = "",
) -> Dict[str, Any]:
    """Treina uma fold do LOSO. Retorna histórico."""

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    smoothing = config.label_smoothing

    for epoch in range(config.max_epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Label smoothing
            if smoothing > 0:
                y_smooth = y_batch * (1 - smoothing) + 0.5 * smoothing
            else:
                y_smooth = y_batch

            logits = model(x_batch)
            loss = criterion(logits, y_smooth)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val)

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Restaurar melhor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    final_epoch = len(history['train_loss'])
    print(f"    [{fold_name}] Stopped at epoch {final_epoch}, "
          f"best val_loss={best_val_loss:.4f}, val_acc={history['val_acc'][-1]:.3f}")

    return history


def run_loso_cv(dataset: GAIADataset, config: LSTMConfig, device: torch.device) -> Dict:
    """
    Leave-One-Subject-Out cross-validation completo.
    Cada fold deixa TODAS as sessões de um participante como teste.
    Retorna resultados por fold e agregados.
    """
    sessions = dataset.get_session_names()

    # ── Mapear sessão → label ──
    session_labels = {}
    for s in sessions:
        idxs = dataset.get_indices_for_sessions([s])
        if idxs:
            session_labels[s] = dataset.labels[idxs[0]]

    # ── Agrupar sessões por participante ──
    participant_sessions: Dict[str, List[str]] = OrderedDict()
    for s in sessions:
        pid = extract_participant_id(s)
        if pid not in participant_sessions:
            participant_sessions[pid] = []
        participant_sessions[pid].append(s)

    participants = list(participant_sessions.keys())
    n_participants = len(participants)

    # Verificar se temos ambas as classes
    classes_present = set(session_labels.values())
    if len(classes_present) < 2:
        print(f"\n  [ERROR] Only {len(classes_present)} class(es) found: {classes_present}")
        print("  Need both NT (0) and TEA (1) for LOSO training.")
        print("  Available sessions:")
        for s, l in sorted(session_labels.items()):
            label_name = "NT" if l == 0 else "TEA"
            n_windows = len(dataset.get_indices_for_sessions([s]))
            print(f"    {s}: {label_name} ({n_windows} windows)")
        return None

    print(f"\n{'='*70}")
    print(f"  LOSO CROSS-VALIDATION — {n_participants} folds (por participante)")
    print(f"{'='*70}")
    for pid in participants:
        sess_list = participant_sessions[pid]
        label = "NT" if session_labels.get(sess_list[0], -1) == 0 else "TEA"
        total_win = sum(len(dataset.get_indices_for_sessions([s])) for s in sess_list)
        print(f"  {pid} ({label}): {len(sess_list)} sessão(ões), {total_win} windows")
        for s in sess_list:
            n_win = len(dataset.get_indices_for_sessions([s]))
            print(f"    └ {s} ({n_win} windows)")
    print(f"{'='*70}\n")

    all_results = {
        'fold_results': [],
        'all_predictions': [],
        'all_labels': [],
        'all_sessions': [],
        'session_predictions': {},
    }

    for fold_idx, test_pid in enumerate(participants):
        test_sessions = participant_sessions[test_pid]
        train_sessions = [s for pid in participants if pid != test_pid
                          for s in participant_sessions[pid]]

        # Verificar se treino tem ambas as classes
        train_labels = set(session_labels[s] for s in train_sessions if s in session_labels)
        if len(train_labels) < 2:
            print(f"  [SKIP] Fold {fold_idx+1} ({test_pid}): "
                  f"training set has only class(es) {train_labels}")
            continue

        train_indices = dataset.get_indices_for_sessions(train_sessions)
        test_indices = dataset.get_indices_for_sessions(test_sessions)

        if not train_indices or not test_indices:
            print(f"  [SKIP] Fold {fold_idx+1}: empty train or test set.")
            continue

        # DataLoaders
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

        pos_weight = dataset.get_class_weights(train_indices)

        train_loader = DataLoader(train_subset, batch_size=config.batch_size,
                                  shuffle=True, drop_last=False)
        test_loader = DataLoader(test_subset, batch_size=config.batch_size,
                                 shuffle=False, drop_last=False)

        # Modelo novo para cada fold
        set_seed(config.seed + fold_idx)
        model = GAIAClassifier(config).to(device)

        fold_name = f"Fold {fold_idx+1}/{n_participants} (test={test_pid})"
        print(f"  ── {fold_name} ──")
        print(f"    Train: {len(train_indices)} windows from {len(train_sessions)} sessions")
        print(f"    Test:  {len(test_indices)} windows from {len(test_sessions)} sessions ({test_pid})")
        for ts in test_sessions:
            lbl = "NT" if session_labels.get(ts, -1) == 0 else "TEA"
            print(f"      {ts} ({lbl})")

        history = train_one_fold(
            model, train_loader, test_loader, config, device,
            pos_weight, fold_name,
        )

        # ── Avaliação por sessão dentro do fold ──
        model.eval()

        # Coletar predições por sessão
        session_window_preds: Dict[str, List[float]] = {s: [] for s in test_sessions}
        session_window_labels: Dict[str, List[float]] = {s: [] for s in test_sessions}
        session_window_attns: Dict[str, List] = {s: [] for s in test_sessions}

        with torch.no_grad():
            for x_batch, y_batch, sid_batch in test_loader:
                x_batch = x_batch.to(device)
                logits, attn_w = model(x_batch, return_attention=True)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                for j, sid in enumerate(sid_batch):
                    session_window_preds[sid].append(float(probs[j]))
                    session_window_labels[sid].append(float(y_batch[j].item()))
                    session_window_attns[sid].append(attn_w[j].cpu().numpy().tolist())

        # Registrar resultado de cada sessão do fold
        for ts in test_sessions:
            preds = session_window_preds[ts]
            labels_ts = session_window_labels[ts]
            attns = session_window_attns[ts]

            if not preds:
                continue

            true_label = session_labels[ts]
            mean_pred = np.mean(preds)
            session_correct = (mean_pred >= 0.5) == (true_label == 1)

            fold_result = {
                'session': ts,
                'participant': test_pid,
                'true_label': true_label,
                'true_label_name': 'TEA' if true_label == 1 else 'NT',
                'mean_prediction': float(mean_pred),
                'std_prediction': float(np.std(preds)),
                'window_predictions': preds,
                'window_labels': labels_ts,
                'attention_weights': attns,
                'session_correct': bool(session_correct),
                'history': history,
            }

            all_results['fold_results'].append(fold_result)
            all_results['all_predictions'].extend(preds)
            all_results['all_labels'].extend(labels_ts)
            all_results['all_sessions'].extend([ts] * len(preds))
            all_results['session_predictions'][ts] = fold_result

            pred_label = "TEA" if mean_pred >= 0.5 else "NT"
            symbol = "✓" if session_correct else "✗"
            print(f"    {ts}: P(TEA)={mean_pred:.3f} → {pred_label} "
                  f"(true: {fold_result['true_label_name']}) {symbol}")

        print()

    return all_results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 6: AVALIAÇÃO & MÉTRICAS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_metrics(results: Dict) -> Dict:
    """Computa métricas globais a partir dos resultados LOSO."""
    preds = np.array(results['all_predictions'])
    labels = np.array(results['all_labels'])

    if len(preds) == 0:
        return {}

    binary_preds = (preds >= 0.5).astype(float)

    tp = np.sum((binary_preds == 1) & (labels == 1))
    tn = np.sum((binary_preds == 0) & (labels == 0))
    fp = np.sum((binary_preds == 1) & (labels == 0))
    fn = np.sum((binary_preds == 0) & (labels == 1))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)  # recall TEA
    specificity = tn / max(tn + fp, 1)  # recall NT
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)

    # AUC-ROC (manual, sem sklearn)
    auc = _compute_auc(labels, preds)

    # Métricas por sessão
    session_results = results.get('fold_results', [])
    session_acc = sum(1 for r in session_results if r['session_correct']) / max(len(session_results), 1)

    return {
        'window_accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'auc_roc': float(auc),
        'session_accuracy': float(session_acc),
        'n_sessions': len(session_results),
        'n_sessions_correct': sum(1 for r in session_results if r['session_correct']),
        'confusion': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
    }


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC via trapezoidal rule (sem dependência do sklearn)."""
    if len(np.unique(labels)) < 2:
        return 0.5

    # Sort by score descending
    desc_idx = np.argsort(-scores)
    sorted_labels = labels[desc_idx]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0

    for label in sorted_labels:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return auc


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 7: MÉTRICAS DESCRITIVAS POR SESSÃO                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_session_descriptives(session_data: Dict) -> Dict:
    """
    Computa métricas descritivas interpretáveis a partir dos dados brutos.
    Estas são as "porcentagens por métrica" do relatório de prognóstico.
    """
    frames = session_data['raw_frames']
    fps = session_data.get('fps_estimate', 30.0)
    features = session_data['features']
    N = len(frames)

    stats = {
        'session_name': session_data['session_name'],
        'total_frames': N,
        'duration_s': round(N / fps, 1),
    }

    # ── Gaze ──
    g_scores = []
    c_scores = []
    g_looking_count = 0
    c_looking_count = 0
    mutual_count = 0
    valid_count = 0

    g_episodes = []
    c_episodes = []
    g_run = 0
    c_run = 0

    for fr in frames:
        g = _get_person(fr, 'GUARDIAN')
        c = _get_person(fr, 'CHILD')
        if g is None or c is None:
            # Fechar episódios se abertos
            if g_run > 0: g_episodes.append(g_run); g_run = 0
            if c_run > 0: c_episodes.append(c_run); c_run = 0
            continue

        valid_count += 1
        g_s = float(g.get('looking_score', 0))
        c_s = float(c.get('looking_score', 0))
        g_scores.append(g_s)
        c_scores.append(c_s)

        g_look = bool(g.get('is_looking', False))
        c_look = bool(c.get('is_looking', False))

        if g_look:
            g_looking_count += 1
            g_run += 1
        else:
            if g_run > 0: g_episodes.append(g_run)
            g_run = 0

        if c_look:
            c_looking_count += 1
            c_run += 1
        else:
            if c_run > 0: c_episodes.append(c_run)
            c_run = 0

        if fr.get('mutual_attention', False):
            mutual_count += 1

    # Fechar últimos episódios
    if g_run > 0: g_episodes.append(g_run)
    if c_run > 0: c_episodes.append(c_run)

    vc = max(valid_count, 1)
    stats['gaze'] = {
        'g_attention_pct': round(100 * g_looking_count / vc, 1),
        'c_attention_pct': round(100 * c_looking_count / vc, 1),
        'mutual_attention_pct': round(100 * mutual_count / vc, 1),
        'g_mean_score': round(float(np.mean(g_scores)) if g_scores else 0.0, 4),
        'c_mean_score': round(float(np.mean(c_scores)) if c_scores else 0.0, 4),
        'g_mean_episode_s': round(np.mean(g_episodes) / fps, 2) if g_episodes else 0.0,
        'c_mean_episode_s': round(np.mean(c_episodes) / fps, 2) if c_episodes else 0.0,
        'g_num_episodes': len(g_episodes),
        'c_num_episodes': len(c_episodes),
    }

    # ── Proximity ──
    valid_mask = ~np.isnan(features[:, 4])
    dist_vals = features[valid_mask, 4]
    dist_vel = features[valid_mask, 5]

    stats['proximity'] = {
        'mean_distance_px': round(float(np.mean(dist_vals)), 1) if len(dist_vals) > 0 else 0,
        'std_distance_px': round(float(np.std(dist_vals)), 1) if len(dist_vals) > 0 else 0,
        'approach_pct': round(100 * np.mean(dist_vel < -0.5), 1) if len(dist_vel) > 0 else 0,
        'withdraw_pct': round(100 * np.mean(dist_vel > 0.5), 1) if len(dist_vel) > 0 else 0,
    }

    # ── Posture / Motor ──
    c_hand = features[valid_mask, 8]
    c_hand_vel = features[valid_mask, 9]
    c_head_mov = features[valid_mask, 13]
    g_shoulder = features[valid_mask, 6]
    c_shoulder = features[valid_mask, 7]

    stats['posture'] = {
        'c_hand_distance_mean': round(float(np.mean(c_hand)), 1) if len(c_hand) > 0 else 0,
        'c_hand_distance_std': round(float(np.std(c_hand)), 1) if len(c_hand) > 0 else 0,
        'c_hand_velocity_std': round(float(np.std(c_hand_vel)), 2) if len(c_hand_vel) > 0 else 0,
        'c_head_movement_mean': round(float(np.mean(c_head_mov)), 2) if len(c_head_mov) > 0 else 0,
        'g_shoulder_tilt_std': round(float(np.std(g_shoulder)), 2) if len(g_shoulder) > 0 else 0,
        'c_shoulder_tilt_std': round(float(np.std(c_shoulder)), 2) if len(c_shoulder) > 0 else 0,
    }

    # ── Temporal ──
    g_streaks = features[valid_mask, 10]
    c_streaks = features[valid_mask, 11]

    stats['temporal'] = {
        'g_max_streak_s': round(float(np.max(g_streaks)) / fps, 2) if len(g_streaks) > 0 else 0,
        'c_max_streak_s': round(float(np.max(c_streaks)) / fps, 2) if len(c_streaks) > 0 else 0,
        'g_mean_streak_when_looking': round(
            float(np.mean(g_streaks[g_streaks > 0])) / fps, 2
        ) if np.any(g_streaks > 0) else 0,
        'c_mean_streak_when_looking': round(
            float(np.mean(c_streaks[c_streaks > 0])) / fps, 2
        ) if np.any(c_streaks > 0) else 0,
    }

    return stats


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 8: RELATÓRIO DE PROGNÓSTICO                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_prognosis_report(
    session_name: str,
    descriptives: Dict,
    prediction: Optional[Dict],
    output_path: str,
):
    """
    Gera relatório de prognóstico legível com métricas percentuais
    por dimensão e score final P(TEA).
    """
    g = descriptives.get('gaze', {})
    p = descriptives.get('proximity', {})
    pos = descriptives.get('posture', {})
    t = descriptives.get('temporal', {})

    tea_prob = prediction.get('mean_prediction', -1) if prediction else -1
    tea_std = prediction.get('std_prediction', 0) if prediction else 0
    tea_ci_lo = max(0, tea_prob - 1.96 * tea_std) if tea_prob >= 0 else -1
    tea_ci_hi = min(1, tea_prob + 1.96 * tea_std) if tea_prob >= 0 else -1

    report = f"""
{'='*70}
  GAIA — RELATÓRIO DE PROGNÓSTICO COMPORTAMENTAL
{'='*70}

  Sessão:    {session_name}
  Duração:   {descriptives.get('duration_s', '?')}s ({descriptives.get('total_frames', '?')} frames)
  Gerado:    {time.strftime('%Y-%m-%d %H:%M:%S')}

{'─'*70}
  1. ENGAJAMENTO VISUAL (GAZE)
{'─'*70}

    Guardian → Child:         {g.get('g_attention_pct', '?')}% dos frames
    Child → Guardian:         {g.get('c_attention_pct', '?')}% dos frames
    Atenção mútua:            {g.get('mutual_attention_pct', '?')}% dos frames

    Score médio Guardian:     {g.get('g_mean_score', '?')}
    Score médio Child:        {g.get('c_mean_score', '?')}

    Episódios Guardian:       {g.get('g_num_episodes', '?')} (média {g.get('g_mean_episode_s', '?')}s)
    Episódios Child:          {g.get('c_num_episodes', '?')} (média {g.get('c_mean_episode_s', '?')}s)

{'─'*70}
  2. PROXIMIDADE
{'─'*70}

    Distância média:          {p.get('mean_distance_px', '?')} px
    Variabilidade (σ):        {p.get('std_distance_px', '?')} px
    Tendência aproximação:    {p.get('approach_pct', '?')}% dos frames
    Tendência afastamento:    {p.get('withdraw_pct', '?')}% dos frames

{'─'*70}
  3. POSTURA E COMPORTAMENTO MOTOR
{'─'*70}

    Mãos da criança (média):  {pos.get('c_hand_distance_mean', '?')} px
    Mãos da criança (σ):      {pos.get('c_hand_distance_std', '?')} px
    Veloc. mãos criança (σ):  {pos.get('c_hand_velocity_std', '?')}
    Movimentação cabeça (C):  {pos.get('c_head_movement_mean', '?')} px/frame
    Inclinação ombro (G) σ:   {pos.get('g_shoulder_tilt_std', '?')}°
    Inclinação ombro (C) σ:   {pos.get('c_shoulder_tilt_std', '?')}°

{'─'*70}
  4. PADRÕES TEMPORAIS
{'─'*70}

    Maior streak G:           {t.get('g_max_streak_s', '?')}s
    Maior streak C:           {t.get('c_max_streak_s', '?')}s
    Streak média G (olhando): {t.get('g_mean_streak_when_looking', '?')}s
    Streak média C (olhando): {t.get('c_mean_streak_when_looking', '?')}s

{'='*70}
  SCORE COMPOSTO — LSTM Bi-direcional + Atenção Temporal
{'='*70}
"""

    if tea_prob >= 0:
        tea_pct = tea_prob * 100
        ci_lo_pct = tea_ci_lo * 100
        ci_hi_pct = tea_ci_hi * 100
        report += f"""
    P(TEA) = {tea_pct:.1f}%
    Intervalo de confiança (95%): [{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]

"""
    else:
        report += """
    P(TEA) = NÃO DISPONÍVEL (modelo não treinado ou sessão não avaliada)

"""

    report += f"""{'='*70}
  NOTA IMPORTANTE
{'='*70}

  Este relatório é gerado por uma ferramenta AUXILIAR de prognóstico
  baseada em visão computacional e redes neurais recorrentes.

  Os resultados NÃO constituem diagnóstico clínico e devem ser
  interpretados EXCLUSIVAMENTE por profissionais qualificados
  (neuropsicólogos, neuropediatras, psiquiatras infantis).

  Ferramenta: GAIA-LSTM v1.0 (TCC — Centro Universitário FEI)
  Pipeline de extração: TITAN v58
{'='*70}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 9: PIPELINE COMPLETA                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_all_sessions(config: LSTMConfig, script_dir: str) -> Tuple[GAIADataset, Dict[str, Dict]]:
    """Carrega todos os JSONs, extrai features, aplica filtro de qualidade, cria dataset."""
    dataset = GAIADataset()
    all_session_data = {}

    print(f"\n{'='*70}")
    print("  GAIA-LSTM — CARREGANDO SESSÕES")
    print(f"{'='*70}\n")

    total_windows = 0
    included_count = 0
    excluded_count = 0

    for group in config.groups:
        json_dir = os.path.join(script_dir, config.output_base, group, "json")
        if not os.path.isdir(json_dir):
            print(f"  [INFO] Diretório não encontrado: {json_dir}")
            continue

        label = config.group_labels.get(group, -1)
        if label < 0:
            print(f"  [WARN] Grupo '{group}' sem label definido, ignorando.")
            continue

        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        label_name = "NT" if label == 0 else "TEA"
        threshold = config.min_valid_pct_nt if label == 0 else config.min_valid_pct_tea

        print(f"  [{group}] ({label_name}) — {len(json_files)} arquivo(s)  "
              f"[limiar qualidade: ≥{threshold:.0f}% frames válidos]")

        for jf in json_files:
            session_data = extract_features_from_json(jf, config)
            if session_data is None:
                continue

            # ── FILTRO DE QUALIDADE: % de frames válidos ──
            pct_valid = session_data['n_valid'] / session_data['n_total'] * 100

            if pct_valid < threshold:
                print(f"    ✗ EXCLUÍDA  {session_data['session_name']}: "
                      f"{pct_valid:.1f}% frames válidos (limiar: {threshold:.0f}%)")
                excluded_count += 1
                continue

            print(f"    ✓ INCLUÍDA  {session_data['session_name']}: "
                  f"{pct_valid:.1f}% frames válidos (limiar: {threshold:.0f}%)")
            included_count += 1

            # Normalizar
            if config.normalize_per_session:
                norm_feats, means, stds = normalize_session(session_data['features'])
            else:
                norm_feats = session_data['features']
                means, stds = None, None

            # Criar janelas
            windows = create_windows(norm_feats, config)
            if not windows:
                print(f"      [WARN] {session_data['session_name']}: 0 janelas válidas, skip.")
                continue

            sname = session_data['session_name']
            session_data['norm_means'] = means
            session_data['norm_stds'] = stds
            all_session_data[sname] = session_data

            # Métricas descritivas
            descriptives = compute_session_descriptives(session_data)
            session_data['descriptives'] = descriptives

            dataset.add_session(sname, windows, label, meta={
                'group': group, 'label': label, 'label_name': label_name,
                'n_frames': session_data['n_total'],
                'n_valid': session_data['n_valid'],
                'pct_valid': round(pct_valid, 1),
                'n_windows': len(windows),
                'participant': extract_participant_id(sname),
            })

            total_windows += len(windows)

    # ── Resumo do filtro ──
    participants = sorted(set(
        extract_participant_id(s) for s in all_session_data.keys()
    ))

    print(f"\n  {'─'*60}")
    print(f"  Filtro de qualidade: {included_count} incluídas, {excluded_count} excluídas")
    print(f"  Total: {len(all_session_data)} sessões, {total_windows} windows")
    print(f"  Participantes: {len(participants)} ({', '.join(participants)})")
    print(f"{'='*70}\n")

    return dataset, all_session_data


def run_full_pipeline(config: LSTMConfig):
    """Pipeline completa: load → train LOSO → evaluate → reports."""

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║            GAIA-LSTM v1.0 — Temporal Behavioral Classifier              ║
║                                                                         ║
║  Ferramenta auxiliar de prognóstico para análise de interação           ║
║  cuidador-criança baseada em padrões temporais de comportamento.        ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    set_seed(config.seed)
    device = get_device()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Diretórios de saída ──
    out_dir = os.path.join(script_dir, config.lstm_output_dir)
    models_dir = os.path.join(out_dir, "models")
    reports_dir = os.path.join(out_dir, "reports")
    results_dir = os.path.join(out_dir, "results")
    for d in [models_dir, reports_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Carregar dados ──
    dataset, all_session_data = load_all_sessions(config, script_dir)

    if len(dataset) == 0:
        print("\n  [ERROR] Nenhuma sessão carregada. Verifique os diretórios de entrada.")
        return

    # ── Gerar relatórios descritivos (independente do treino) ──
    print(f"\n{'='*70}")
    print("  GERANDO RELATÓRIOS DESCRITIVOS")
    print(f"{'='*70}\n")

    for sname, sdata in all_session_data.items():
        desc = sdata.get('descriptives', {})
        report_path = os.path.join(reports_dir, f"descritivo_{sname}.txt")
        generate_prognosis_report(sname, desc, None, report_path)
        print(f"  ✓ {report_path}")

    # ── Treinar com LOSO ──
    sessions = dataset.get_session_names()
    session_labels_set = set()
    for s in sessions:
        idxs = dataset.get_indices_for_sessions([s])
        if idxs:
            session_labels_set.add(dataset.labels[idxs[0]])

    if len(session_labels_set) < 2:
        print(f"\n  [INFO] Apenas {len(session_labels_set)} classe(s) disponível(is).")
        print("  Relatórios descritivos foram gerados.")
        print("  Para treinar o LSTM, são necessárias sessões de AMBOS os grupos (NT + TEA).")
        print(f"\n  Sessões carregadas:")
        for sname, sdata in all_session_data.items():
            meta = dataset.session_meta.get(sname, {})
            print(f"    {sname}: {meta.get('label_name', '?')} "
                  f"({meta.get('n_windows', 0)} windows)")
        return

    results = run_loso_cv(dataset, config, device)

    if results is None:
        print("  [ERROR] LOSO falhou. Verifique os dados.")
        return

    # ── Métricas globais ──
    metrics = compute_metrics(results)

    print(f"\n{'='*70}")
    print("  RESULTADOS FINAIS — LOSO CROSS-VALIDATION")
    print(f"{'='*70}\n")

    cm = metrics.get('confusion', {})
    print(f"  Session-level accuracy: {metrics['session_accuracy']*100:.1f}% "
          f"({metrics['n_sessions_correct']}/{metrics['n_sessions']})")
    print(f"  Window-level accuracy:  {metrics['window_accuracy']*100:.1f}%")
    print(f"  Sensitivity (TEA):      {metrics['sensitivity']*100:.1f}%")
    print(f"  Specificity (NT):       {metrics['specificity']*100:.1f}%")
    print(f"  Precision:              {metrics['precision']*100:.1f}%")
    print(f"  F1 Score:               {metrics['f1']*100:.1f}%")
    print(f"  AUC-ROC:                {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix (windows):")
    print(f"                    Predicted NT    Predicted TEA")
    print(f"    Actual NT       {cm.get('tn',0):>8}        {cm.get('fp',0):>8}")
    print(f"    Actual TEA      {cm.get('fn',0):>8}        {cm.get('tp',0):>8}")

    # ── Resultados por sessão ──
    print(f"\n  {'─'*60}")
    print(f"  {'Sessão':<30} {'True':>6} {'P(TEA)':>8} {'Pred':>6} {'OK':>4}")
    print(f"  {'─'*60}")
    for fr in results['fold_results']:
        pred_label = "TEA" if fr['mean_prediction'] >= 0.5 else "NT"
        symbol = "✓" if fr['session_correct'] else "✗"
        print(f"  {fr['session']:<30} {fr['true_label_name']:>6} "
              f"{fr['mean_prediction']:>8.3f} {pred_label:>6} {symbol:>4}")
    print(f"  {'─'*60}")

    # ── Gerar relatórios completos (com P(TEA)) ──
    print(f"\n  Gerando relatórios com score LSTM...")
    for fr in results['fold_results']:
        sname = fr['session']
        if sname in all_session_data:
            desc = all_session_data[sname].get('descriptives', {})
            report_path = os.path.join(reports_dir, f"prognostico_{sname}.txt")
            generate_prognosis_report(sname, desc, fr, report_path)
            print(f"    ✓ {report_path}")

    # ── Salvar resultados como JSON ──
    results_save = {
        'config': {
            'window_size': config.window_size,
            'window_stride': config.window_stride,
            'hidden_dim': config.hidden_dim,
            'num_lstm_layers': config.num_lstm_layers,
            'dropout': config.dropout,
            'bidirectional': config.bidirectional,
            'learning_rate': config.learning_rate,
            'max_epochs': config.max_epochs,
            'seed': config.seed,
            'min_valid_pct_nt': config.min_valid_pct_nt,
            'min_valid_pct_tea': config.min_valid_pct_tea,
        },
        'metrics': metrics,
        'session_results': [
            {
                'session': fr['session'],
                'participant': fr.get('participant', extract_participant_id(fr['session'])),
                'true_label': fr['true_label_name'],
                'p_tea': round(fr['mean_prediction'], 4),
                'p_tea_std': round(fr['std_prediction'], 4),
                'correct': fr['session_correct'],
            }
            for fr in results['fold_results']
        ],
    }

    results_path = os.path.join(results_dir, "loso_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_save, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Resultados salvos: {results_path}")

    # ── Salvar modelo treinado no dataset completo (para inferência futura) ──
    print(f"\n  Treinando modelo final em todo o dataset...")
    set_seed(config.seed)
    final_model = GAIAClassifier(config).to(device)
    all_indices = list(range(len(dataset)))
    pos_weight = dataset.get_class_weights(all_indices)

    full_loader = DataLoader(dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=False)
    # Usar um loader "dummy" como validação (o mesmo treino — só para early stopping da loss)
    train_one_fold(final_model, full_loader, full_loader, config, device,
                   pos_weight, "Final model")

    model_path = os.path.join(models_dir, "gaia_lstm_final.pt")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': config.__dict__,
        'metrics': metrics,
    }, model_path)
    print(f"  ✓ Modelo salvo: {model_path}")

    print(f"\n{'='*70}")
    print("  GAIA-LSTM PIPELINE COMPLETA")
    print(f"{'='*70}\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 10: INFERÊNCIA EM NOVA SESSÃO                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def predict_session(json_path: str, model_path: str, config: LSTMConfig = None):
    """
    Roda inferência em uma sessão nova usando modelo treinado.
    Gera relatório de prognóstico.
    """
    if config is None:
        config = LSTMConfig()

    print(f"\n{'='*70}")
    print(f"  GAIA-LSTM — INFERÊNCIA")
    print(f"{'='*70}\n")

    device = get_device()

    # Carregar modelo
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', {})

    # Reconstruir config do checkpoint se disponível
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = GAIAClassifier(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extrair features
    session_data = extract_features_from_json(json_path, config)
    if session_data is None:
        print("  [ERROR] Falha ao extrair features.")
        return None

    # Normalizar
    norm_feats, _, _ = normalize_session(session_data['features'])

    # Criar janelas
    windows = create_windows(norm_feats, config)
    if not windows:
        print("  [ERROR] 0 janelas válidas.")
        return None

    # Inferência
    predictions = []
    attention_weights = []

    with torch.no_grad():
        for w in windows:
            x = torch.FloatTensor(w).unsqueeze(0).to(device)
            logits, attn_w = model(x, return_attention=True)
            prob = torch.sigmoid(logits).item()
            predictions.append(prob)
            attention_weights.append(attn_w.cpu().numpy().flatten())

    mean_pred = float(np.mean(predictions))
    std_pred = float(np.std(predictions))

    prediction_result = {
        'mean_prediction': mean_pred,
        'std_prediction': std_pred,
        'window_predictions': predictions,
        'attention_weights': [aw.tolist() for aw in attention_weights],
    }

    # Métricas descritivas
    descriptives = compute_session_descriptives(session_data)

    # Gerar relatório
    sname = session_data['session_name']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, config.lstm_output_dir, "reports",
                               f"prognostico_{sname}.txt")
    report = generate_prognosis_report(sname, descriptives, prediction_result, report_path)
    print(report)
    print(f"  ✓ Relatório salvo: {report_path}")

    return prediction_result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 11: ENTRY POINT                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GAIA-LSTM v1.0 — Temporal Behavioral Pattern Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Treinar com LOSO cross-validation (pipeline completa):
  python gaia_lstm_v1.py

  # Inferência em uma sessão nova:
  python gaia_lstm_v1.py --predict path/to/json_session.json --model lstm_output/models/gaia_lstm_final.pt

  # Customizar hiperparâmetros:
  python gaia_lstm_v1.py --hidden 256 --layers 3 --dropout 0.4 --window 450 --lr 0.0005
        """,
    )

    parser.add_argument('--predict', type=str, default=None,
                        help='JSON path for inference on a new session')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (.pt) for inference')
    parser.add_argument('--hidden', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--window', type=int, default=300,
                        help='Window size in frames (default: 300 = ~10s)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Window stride (default: window_size // 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Max training epochs (default: 150)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    args = parser.parse_args()

    config = LSTMConfig(
        hidden_dim=args.hidden,
        num_lstm_layers=args.layers,
        dropout=args.dropout,
        window_size=args.window,
        window_stride=args.stride if args.stride else args.window // 2,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed,
        patience=args.patience,
    )

    if args.predict:
        if not args.model:
            print("[ERROR] --model é obrigatório para inferência.")
            print("  Ex: python gaia_lstm_v1.py --predict session.json --model lstm_output/models/gaia_lstm_final.pt")
            sys.exit(1)
        predict_session(args.predict, args.model, config)
    else:
        run_full_pipeline(config)