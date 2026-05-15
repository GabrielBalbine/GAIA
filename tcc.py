"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║          ████████╗██╗████████╗ █████╗ ███╗   ██╗    ██╗   ██╗██████╗  █████╗  ║
║          ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║    ██║   ██║╚════██╗██╔══██╗ ║
║             ██║   ██║   ██║   ███████║██╔██╗ ██║    ██║   ██║ █████╔╝╚█████╔╝ ║
║             ██║   ██║   ██║   ██╔══██║██║╚██╗██║    ╚██╗ ██╔╝██╔═══╝ ██╔══██╗ ║
║             ██║   ██║   ██║   ██║  ██║██║ ╚████║     ╚████╔╝ ███████╗╚█████╔╝ ║
║             ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝      ╚═══╝  ╚══════╝ ╚════╝  ║
║                                                                               ║
║                    ★★★ GAIA — Gaze Analysis for Interaction Assessment ★★★   ║
║                                                                               ║
║  FILOSOFIA:                                                                   ║
║   • SÓ produz dados quando AMBAS as pessoas estão detectadas                  ║
║   • Zero falso positivo: prefere silêncio a dado errado                       ║
║   • Anti-falso-positivo: filtro rigoroso de humano real                       ║
║   • MediaPipe Face Mesh 2D + ÍRIS (468-477) + YOLO keypoints combinados      ║
║   • Dados de linguagem corporal integrados                                    ║
║                                                                               ║
║  OTIMIZAÇÕES v53 (sweet spot final):                                         ║
║   • IRIS GAZE: usa landmarks 468-477 (íris real) com peso dinâmico           ║
║     por frontalidade — frontview: íris 55%, face 45% / sideview: face 100%  ║
║   • DEQUE ACCUMULATION: janela deslizante (maxlen=90) sem restart manual     ║
║     — lock recusado não joga dados fora, só deixa deque substituir           ║
║   • HONEYMOON POST-LOCK: 60 frames após lock, monitora inversão silenciosa   ║
║     — se >45% dos frames mostram CHILD > GUARDIAN → lock errado → reset     ║
║   • APPEARANCE DESEMPATE: histograma HSV durante acumulação pré-lock         ║
║     — candidatos claramente distintos em cor → threshold de ratio mais leve  ║
║   • Modelo yolov8l-pose @ 960px                                              ║
║   • Multi-pass condicional, aparência throttled, JSON streaming              ║
║   • ThreadedVideoWriter, batch com processed.txt                             ║
║                                                                               ║
║  MELHORIAS v58 (similar-size pair robustness):                               ║
║   1. Lock threshold ADAPTATIVO: ratio<1.35 → score≥0.68, margin≥0.06       ║
║   2. Honeymoon ignora bad_rate puro em pares similares (só swap_rate conta) ║
║   3. Floor penalties PROPORCIONAIS ao prescan ratio (ratio_factor)           ║
║   4. Prescan valida consistência sw — flag sw_reliable, peso reduzido       ║
║   5. Re-lock ACELERADO: memória viva → amostras_para_decisao/2             ║
║   6. Anti-cascata: grace period 30f para pair-ambiguous cooldown            ║
║                                                                               ║
║  BLINDAGEM v53 (tudo que já tinha + honeymoon):                              ║
║   • Skip 60s, ghost IDs negativos, size guard FLOOR/CEILING                 ║
║   • TRUSTED shoulder minimum (30px), consistency buffer (15 frames)         ║
║   • Memory freeze durante mismatch, global matching anti-duplicação         ║
║   • 3-person cooldown anti-pesquisador, honeymoon anti-lock-errado           ║
║                                                                               ║
║  REGRA DE OURO: Se não encontrou 2 pessoas → frame descartado.               ║
║                 Dado ruim é PIOR que dado nenhum.                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import json
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
os.environ['PYTHONUNBUFFERED'] = '1'

import time
import torch
import threading
import queue

try:
    _stdout_bak = sys.stdout
    _stderr_bak = sys.stderr
    _devnull = None
    try:
        _devnull = open(os.devnull, 'w')
        sys.stderr = _devnull
    except Exception:
        pass
    import mediapipe as mp
    sys.stderr = _stderr_bak if _stderr_bak and not _stderr_bak.closed else sys.__stderr__
    sys.stdout = _stdout_bak if _stdout_bak and not _stdout_bak.closed else sys.__stdout__
    if _devnull is not None:
        try:
            _devnull.close()
        except Exception:
            pass
    MEDIAPIPE_AVAILABLE = True
except Exception:
    sys.stdout = getattr(sys, '__stdout__', sys.stdout)
    sys.stderr = getattr(sys, '__stderr__', sys.stderr)
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: mediapipe not available. Using YOLO-only for gaze.")

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any
from tqdm import tqdm
from ultralytics import YOLO


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 0: DIAGNÓSTICO DE GPU                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def diagnose_gpu() -> dict:
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cpu',
        'device_name': 'CPU',
        'gpu_count': 0,
        'vram_total_gb': 0,
        'vram_free_gb': 0,
        'cuda_version': 'N/A',
        'cudnn_version': 'N/A',
        'cudnn_enabled': False,
        'fp16_supported': False,
        'bf16_supported': False,
        'torch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        info['device'] = 'cuda:0'
        info['gpu_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda or 'N/A'
        info['cudnn_version'] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else 'N/A'
        info['cudnn_enabled'] = torch.backends.cudnn.enabled

        props = torch.cuda.get_device_properties(0)
        vram_total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        if hasattr(torch.cuda, 'mem_get_info'):
            vram_free_real, vram_total_real = torch.cuda.mem_get_info(0)
            info['vram_total_gb'] = round(vram_total_real / (1024**3), 2)
            info['vram_free_gb'] = round(vram_free_real / (1024**3), 2)
        else:
            vram_free = vram_total - torch.cuda.memory_allocated(0)
            info['vram_total_gb'] = round(vram_total / (1024**3), 2)
            info['vram_free_gb'] = round(vram_free / (1024**3), 2)
        info['vram_allocated_mb'] = round(torch.cuda.memory_allocated(0) / (1024**2))
        info['vram_reserved_mb'] = round(torch.cuda.memory_reserved(0) / (1024**2))

        compute_cap = (props.major, props.minor)
        info['compute_capability'] = f"{props.major}.{props.minor}"
        info['fp16_supported'] = compute_cap >= (5, 3)
        info['bf16_supported'] = compute_cap >= (8, 0)

    return info


def print_gpu_banner(info: dict):
    print("\n" + "=" * 70)
    print("  DIAGNÓSTICO DE GPU")
    print("=" * 70)

    if info['cuda_available']:
        print(f"  CUDA:      AVAILABLE (v{info['cuda_version']})")
        print(f"  GPU:       {info['device_name']}")
        print(f"  VRAM:      {info['vram_total_gb']} GB total, {info['vram_free_gb']} GB livre (driver)")
        if 'vram_allocated_mb' in info:
            print(f"  PyTorch:   {info['vram_allocated_mb']} MB allocated, {info['vram_reserved_mb']} MB reserved")
        print(f"  Compute:   {info.get('compute_capability', '?')}")
        print(f"  cuDNN:     {'v' + info['cudnn_version'] if info['cudnn_enabled'] else 'DESABILITADO'}")
        print(f"  FP16:      {'YES' if info['fp16_supported'] else 'NO (old GPU)'}")
        print(f"  BF16:      {'YES' if info['bf16_supported'] else 'NO'}")
        print(f"  PyTorch:   {info['torch_version']}")
        print(f"  Device:    {info['device']}")
        if info['fp16_supported']:
            print("\n  >>> FP16 (half precision) ENABLED — 2x faster, same accuracy")
        if info['cudnn_enabled']:
            print("  >>> cuDNN benchmark ENABLED — auto-optimizes kernels")
    else:
        print("  CUDA:      NOT AVAILABLE")
        print("  PyTorch:   " + info['torch_version'])
        print("\n  WARNING: Running on CPU. Will be MUCH slower.")

    print("=" * 70 + "\n")


def maximize_cpu():
    n_cores = os.cpu_count() or 4
    cv2.setNumThreads(0)
    torch.set_num_threads(n_cores)
    try:
        torch.set_num_interop_threads(max(1, n_cores // 2))
    except RuntimeError:
        pass
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[var] = str(n_cores)
    return n_cores


class ThreadedVideoWriter:
    def __init__(self, path: str, fourcc: int, fps: int, size: tuple, buffer_size: int = 64):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def _writer_loop(self):
        while self.running or not self.buffer.empty():
            try:
                frame = self.buffer.get(timeout=1)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def write(self, frame: np.ndarray):
        self.buffer.put(frame)

    def release(self):
        self.running = False
        self.thread.join(timeout=30)
        self.writer.release()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 1: CONFIGURAÇÃO CENTRALIZADA                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TitanConfig:
    # ── Caminhos ──
    input_video: str = "video_teste_2_pessoas.mp4"
    output_video: str = "output_TITAN_v58_GAIA.mp4"
    output_json: str = "dados_TITAN_v58_GAIA.json"
    output_report: str = "relatorio_TITAN_v58_GAIA.txt"

    # ── Modelo ──
    model_name: str = "yolov8l-pose.pt"

    # ── GPU / CUDA ──
    device: str = "auto"
    use_fp16: bool = True
    cudnn_benchmark: bool = True
    imgsz: int = 960
    use_tta: bool = False

    # ── Performance / estabilidade ──
    appearance_update_interval: int = 5
    mediapipe_min_yolo_face_conf: float = 0.45
    candidate_pool_max: int = 5
    min_box_conf_tracked: float = 0.18
    min_box_conf_extra: float = 0.10

    # ── Detecção Multi-Pass ──
    detection_passes: list = field(default_factory=lambda: [
        {"conf": 0.30, "iou": 0.65},
        {"conf": 0.18, "iou": 0.55},
        {"conf": 0.10, "iou": 0.45},
    ])

    # ── Heurística de humano real ──
    human_score_strict: float = 0.62
    human_score_trusted: float = 0.48
    human_score_extra_pass: float = 0.58
    hard_min_bbox_height: float = 70.0
    hard_min_bbox_height_if_small: float = 95.0
    hard_min_size_signature_if_small: float = 82.0
    min_pair_separation_px: float = 55.0

    # ── Validação de Keypoints ──
    min_kpt_visiveis: int = 4
    min_conf_media_kpt: float = 0.12

    # ── Timing ──
    segundos_para_pular: float = 60.0
    prescan_stride_frames: int = 15

    # ── Identificação / Robustez de Papéis ──
    amostras_para_decisao: int = 45
    pontos_tronco: list = field(default_factory=lambda: [5, 6, 11, 12])
    peso_altura_tronco: float = 0.30
    peso_largura_ombros: float = 0.35
    peso_bbox_area: float = 0.20
    peso_bbox_height: float = 0.15
    role_fit_accept_threshold: float = 0.52
    role_fit_margin: float = 0.08
    lock_pair_min_score: float = 1.12
    lock_pair_min_margin: float = 0.18
    guardian_min_height_ratio_to_prescan: float = 0.58
    guardian_min_size_ratio_to_prescan: float = 0.60
    child_min_height_ratio_to_prescan: float = 0.52
    child_min_size_ratio_to_prescan: float = 0.58
    pair_min_size_ratio_to_prescan: float = 0.72
    pair_min_bh_ratio_to_prescan: float = 0.70
    inversion_suspect_margin: float = 0.10
    inversion_reset_margin: float = 0.18
    honeymoon_low_score_threshold: float = 1.12
    honeymoon_bad_vote_threshold: float = 0.25
    identity_stale_frames: int = 90
    role_ban_frames: int = 120
    role_uncertainty_cooldown_frames: int = 18
    recovery_min_track_hits: int = 4
    recovery_min_track_trust: float = 0.42
    enable_direct_role_swap: bool = False

    # ── Gaze / Looking Detection ──
    looking_threshold_enter: float = 0.28
    looking_threshold_exit: float = 0.20
    ema_alpha: float = 0.28
    voting_window_size: int = 7
    camera_y_weight: float = 0.3

    # ── Memória Persistente ──
    memory_max_frames: int = 300
    appearance_history_size: int = 120
    hist_bins: int = 64
    reid_appearance_threshold: float = 0.35
    reid_weight_appearance: float = 0.35
    reid_weight_position: float = 0.35
    reid_weight_size: float = 0.30

    # ── Predição de Posição ──
    max_frames_predicao: int = 60

    # ── Regra de Ouro ──
    strict_two_person: bool = True

    # ── Visualização ──
    draw_skeleton: bool = True
    draw_gaze_vector: bool = True
    draw_connection_line: bool = True
    draw_dashboard: bool = True
    draw_memory_ghosts: bool = True
    gaze_vector_magnitude: float = 80.0

    # ── Privacidade ──
    privacy_blur: bool = True
    privacy_blur_kernel: int = 51

    # ── Re-ID / Detecção ──
    reid_max_distance: int = 400
    duplicate_distance: int = 100
    position_norm_sigma: float = 200.0

    # ── JSON ──
    json_indent: int = 2
    json_flush_interval: int = 25

    # ── Logging / terminal ──
    console_debug_events: bool = False
    console_identity_events: bool = False
    console_gaze_errors: bool = False
    console_gaze_error_cooldown_frames: int = 120
    debug_log_flush_immediate: bool = False

    # ── Skeleton (COCO format) ──
    skeleton_connections: list = field(default_factory=lambda: [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ])

    def __post_init__(self):
        errors = []
        warns = []
        if self.privacy_blur and (self.privacy_blur_kernel < 3 or self.privacy_blur_kernel % 2 == 0):
            errors.append(f"privacy_blur_kernel deve ser ímpar e >= 3, recebeu {self.privacy_blur_kernel}")
        if not (0.0 <= self.camera_y_weight <= 1.0):
            errors.append(f"camera_y_weight deve ser [0, 1], recebeu {self.camera_y_weight}")
        if self.appearance_update_interval < 1:
            errors.append(f"appearance_update_interval deve ser >= 1, recebeu {self.appearance_update_interval}")
        if self.looking_threshold_enter <= self.looking_threshold_exit:
            errors.append("looking_threshold_enter deve ser > looking_threshold_exit")
        if self.voting_window_size < 1:
            errors.append(f"voting_window_size deve ser >= 1, recebeu {self.voting_window_size}")
        if not (0.0 < self.ema_alpha <= 1.0):
            errors.append(f"ema_alpha deve ser (0, 1], recebeu {self.ema_alpha}")
        if not (0.0 < self.human_score_trusted <= self.human_score_strict <= 1.0):
            errors.append("human_score_trusted / human_score_strict inválidos")
        if not (0.0 < self.human_score_extra_pass <= 1.0):
            errors.append("human_score_extra_pass inválido")
        if self.hard_min_bbox_height < 0 or self.hard_min_bbox_height_if_small < 0 or self.hard_min_size_signature_if_small < 0:
            errors.append("hard_min_* thresholds inválidos")
        if not (0.0 < self.role_fit_accept_threshold <= 1.0):
            errors.append("role_fit_accept_threshold inválido")
        if self.role_fit_margin <= 0:
            errors.append("role_fit_margin deve ser > 0")
        if self.lock_pair_min_margin <= 0:
            errors.append("lock_pair_min_margin deve ser > 0")
        for name in [
            'guardian_min_height_ratio_to_prescan', 'guardian_min_size_ratio_to_prescan',
            'child_min_height_ratio_to_prescan', 'child_min_size_ratio_to_prescan',
            'pair_min_size_ratio_to_prescan', 'pair_min_bh_ratio_to_prescan',
            'honeymoon_bad_vote_threshold'
        ]:
            val = getattr(self, name)
            if not (0.0 < val <= 1.0):
                errors.append(f"{name} inválido")
        if self.inversion_suspect_margin <= 0 or self.inversion_reset_margin <= self.inversion_suspect_margin:
            errors.append("inversion_* margins inválidos")
        if self.honeymoon_low_score_threshold <= 0:
            errors.append("honeymoon_low_score_threshold inválido")
        if self.candidate_pool_max < 2:
            errors.append("candidate_pool_max deve ser >= 2")
        if self.identity_stale_frames < 1:
            errors.append("identity_stale_frames deve ser >= 1")
        if self.role_ban_frames < 1:
            errors.append("role_ban_frames deve ser >= 1")
        if self.role_uncertainty_cooldown_frames < 0:
            errors.append("role_uncertainty_cooldown_frames deve ser >= 0")
        if self.recovery_min_track_hits < 1:
            errors.append("recovery_min_track_hits deve ser >= 1")
        if not (0.0 <= self.recovery_min_track_trust <= 1.0):
            errors.append("recovery_min_track_trust inválido")
        if self.json_indent is not None and self.json_indent < 0:
            errors.append("json_indent deve ser >= 0")
        if self.console_gaze_error_cooldown_frames < 1:
            errors.append("console_gaze_error_cooldown_frames deve ser >= 1")
        for i, p in enumerate(self.detection_passes):
            if not (0.0 < p.get('conf', 0) <= 1.0):
                errors.append(f"detection_passes[{i}].conf inválido")
            if not (0.0 < p.get('iou', 0) <= 1.0):
                errors.append(f"detection_passes[{i}].iou inválido")
        if self.imgsz % 32 != 0:
            warns.append(f"imgsz={self.imgsz} is not a multiple of 32")
        if not os.path.isfile(self.input_video):
            errors.append(f"Input video not found: '{self.input_video}'")
        for w in warns:
            print(f"  [WARN] {w}")
        if errors:
            for e in errors:
                print(f"  [ERROR] {e}")
            raise ValueError(f"TitanConfig invalid: {len(errors)} error(s) found")


def metric_closeness(value: Optional[float], ref: Optional[float], tolerance: float = 0.45) -> Optional[float]:
    if value is None or ref is None or ref <= 0:
        return None
    rel = abs(float(value) - float(ref)) / max(float(ref), 1.0)
    return float(max(0.0, 1.0 - rel / max(tolerance, 1e-6)))


def relative_order_score(
    larger: Optional[float],
    smaller: Optional[float],
    ambiguity: float = 0.03,
    saturation: float = 0.30,
) -> Optional[float]:
    if larger is None or smaller is None:
        return None
    larger = float(larger)
    smaller = float(smaller)
    denom = max(max(abs(larger), abs(smaller)), 1.0)
    rel = (larger - smaller) / denom
    low = -abs(float(ambiguity))
    high = max(abs(float(saturation)), low + 1e-6)
    score = (rel - low) / max(high - low, 1e-6)
    return float(max(0.0, min(1.0, score)))


def safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    b = float(b)
    if abs(b) < 1e-6:
        return None
    return float(a) / b


def bbox_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def body_metrics_from_pose(kpts: np.ndarray, box: Optional[np.ndarray] = None) -> Dict[str, Any]:
    conf = kpts[:, 2] if kpts is not None and len(kpts) >= 17 else np.zeros(17, dtype=np.float32)
    visible_mask = conf > 0.08
    visible_pts = kpts[visible_mask][:, :2] if np.any(visible_mask) else np.empty((0, 2), dtype=np.float32)
    vis_count = int(np.sum(visible_mask))
    mean_conf = float(np.mean(conf[visible_mask])) if vis_count else 0.0

    head_ids = [0, 1, 2, 3, 4]
    shoulder_ids = [5, 6]
    hip_ids = [11, 12]
    lower_ids = [13, 14, 15, 16]
    upper_ids = [5, 6, 7, 8, 9, 10, 11, 12]

    head_count = int(sum(conf[i] > 0.08 for i in head_ids))
    shoulders_count = int(sum(conf[i] > 0.08 for i in shoulder_ids))
    hips_count = int(sum(conf[i] > 0.08 for i in hip_ids))
    lower_count = int(sum(conf[i] > 0.08 for i in lower_ids))
    upper_count = int(sum(conf[i] > 0.08 for i in upper_ids))

    shoulder_width = None
    if conf[5] > 0.08 and conf[6] > 0.08:
        shoulder_width = float(np.linalg.norm(kpts[6][:2] - kpts[5][:2]))

    hip_width = None
    if conf[11] > 0.08 and conf[12] > 0.08:
        hip_width = float(np.linalg.norm(kpts[12][:2] - kpts[11][:2]))

    torso_height = None
    if conf[5] > 0.08 and conf[6] > 0.08 and conf[11] > 0.08 and conf[12] > 0.08:
        mid_sh = (kpts[5][:2] + kpts[6][:2]) / 2.0
        mid_hip = (kpts[11][:2] + kpts[12][:2]) / 2.0
        torso_height = float(np.linalg.norm(mid_hip - mid_sh))

    if box is not None:
        bbox_width = float(max(0.0, box[2] - box[0]))
        bbox_height = float(max(0.0, box[3] - box[1]))
        bbox_area = float(bbox_width * bbox_height)
    elif len(visible_pts) > 0:
        xs = visible_pts[:, 0]
        ys = visible_pts[:, 1]
        bbox_width = float(np.ptp(xs))
        bbox_height = float(np.ptp(ys))
        bbox_area = float(bbox_width * bbox_height)
    else:
        bbox_width = 0.0
        bbox_height = 0.0
        bbox_area = 0.0

    head_y = None
    head_visible = [float(kpts[i][1]) for i in head_ids if conf[i] > 0.08]
    if head_visible:
        head_y = float(np.median(head_visible))

    shoulders_y = None
    shoulder_visible = [float(kpts[i][1]) for i in shoulder_ids if conf[i] > 0.08]
    if shoulder_visible:
        shoulders_y = float(np.mean(shoulder_visible))

    head_above_shoulders = None
    if head_y is not None and shoulders_y is not None:
        head_above_shoulders = bool(head_y < shoulders_y)

    if len(visible_pts) > 0:
        center = np.mean(visible_pts, axis=0)
    elif conf[0] > 0.05:
        center = kpts[0][:2].copy()
    else:
        center = np.zeros(2, dtype=np.float32)

    center_y = float(center[1]) if center is not None else 0.0
    aspect_ratio = float(bbox_height / max(bbox_width, 1.0)) if bbox_height > 0 else 0.0

    size_terms = []
    if shoulder_width is not None:
        size_terms.append((shoulder_width, 0.35))
    if torso_height is not None:
        size_terms.append((torso_height, 0.30))
    if bbox_height > 0:
        size_terms.append((bbox_height, 0.20))
    if bbox_area > 0:
        size_terms.append((np.sqrt(bbox_area), 0.15))
    size_signature = float(sum(v * w for v, w in size_terms)) if size_terms else 0.0

    return {
        'vis_count': vis_count,
        'mean_conf': mean_conf,
        'head_count': head_count,
        'shoulders_count': shoulders_count,
        'hips_count': hips_count,
        'lower_count': lower_count,
        'upper_count': upper_count,
        'shoulder_width': shoulder_width,
        'hip_width': hip_width,
        'torso_height': torso_height,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'bbox_area': bbox_area,
        'aspect_ratio': aspect_ratio,
        'head_above_shoulders': head_above_shoulders,
        'center': center.astype(np.float32) if isinstance(center, np.ndarray) else np.array(center, dtype=np.float32),
        'center_y': center_y,
        'size_signature': size_signature,
        'core_torso': bool(shoulders_count == 2 and hips_count >= 1),
        'partial_upper_body': bool(shoulders_count >= 1 and head_count >= 1 and upper_count >= 4),
    }


def score_human_pose(
    kpts: np.ndarray,
    box: Optional[np.ndarray] = None,
    box_conf: Optional[float] = None,
    mode: str = 'strict',
    config: Optional[TitanConfig] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    metrics = body_metrics_from_pose(kpts, box)

    box_conf_val = float(box_conf) if box_conf is not None else 0.35
    score = 0.0
    score += min(metrics['vis_count'] / 10.0, 1.0) * 0.16
    score += min(metrics['mean_conf'] / 0.60, 1.0) * 0.10
    score += min(max(box_conf_val, 0.0) / 0.60, 1.0) * 0.08

    if metrics['shoulders_count'] == 2:
        score += 0.16
    elif metrics['shoulders_count'] == 1:
        score += 0.06

    if metrics['hips_count'] == 2:
        score += 0.10
    elif metrics['hips_count'] == 1:
        score += 0.06

    if metrics['head_count'] >= 1 and metrics['head_above_shoulders'] is True:
        score += 0.10
    elif metrics['head_count'] >= 1:
        score += 0.03

    if metrics['lower_count'] >= 2:
        score += 0.08
    elif metrics['lower_count'] >= 1:
        score += 0.05
    elif metrics['bbox_height'] >= 140 and metrics['upper_count'] >= 4:
        score += 0.03

    if metrics['torso_height'] is not None and metrics['shoulder_width'] is not None:
        ratio = metrics['torso_height'] / max(metrics['shoulder_width'], 1.0)
        if 0.45 <= ratio <= 3.00:
            score += 0.18
        elif 0.30 <= ratio <= 4.00:
            score += 0.10
        else:
            score -= 0.12

    if 0.45 <= metrics['aspect_ratio'] <= 5.50:
        score += 0.07
    elif metrics['aspect_ratio'] < 0.28 or metrics['aspect_ratio'] > 6.50:
        score -= 0.12

    if metrics['bbox_height'] >= max(55.0, (metrics['shoulder_width'] or 0.0) * 1.35):
        score += 0.07

    if metrics['shoulder_width'] is not None:
        min_sw = max(16.0, metrics['bbox_height'] * 0.07)
        if metrics['shoulder_width'] < min_sw:
            score -= 0.10

    support = metrics['core_torso'] or metrics['partial_upper_body'] or (
        metrics['shoulders_count'] == 2 and metrics['upper_count'] >= 4 and metrics['bbox_height'] >= 90
    )
    if not support:
        score -= 0.20

    if metrics['head_above_shoulders'] is False:
        score -= 0.12

    if metrics['bbox_height'] < 45 or metrics['bbox_area'] < 2500:
        metrics['human_score'] = 0.0
        metrics['reject_reason'] = 'too_small_absolute'
        return False, 0.0, metrics

    tiny_combo = False
    if config is not None:
        tiny_combo = (
            metrics['bbox_height'] < config.hard_min_bbox_height_if_small and
            metrics['size_signature'] < config.hard_min_size_signature_if_small
        )
        if metrics['bbox_height'] < config.hard_min_bbox_height:
            metrics['human_score'] = 0.0
            metrics['reject_reason'] = 'below_hard_min_bbox_height'
            return False, 0.0, metrics
        if tiny_combo:
            metrics['human_score'] = 0.0
            metrics['reject_reason'] = 'tiny_blob_pose'
            return False, 0.0, metrics

    if metrics['bbox_height'] < 120:
        score -= 0.12
    if metrics['size_signature'] < 95:
        score -= 0.08
    if metrics['bbox_height'] < 135 and not metrics['core_torso']:
        score -= 0.10

    score = float(max(0.0, min(1.0, score)))

    if config is not None:
        if mode == 'trusted':
            threshold = config.human_score_trusted
        elif mode == 'extra':
            threshold = config.human_score_extra_pass
        else:
            threshold = config.human_score_strict
    else:
        threshold = 0.62 if mode == 'strict' else 0.48

    accept = bool(score >= threshold and support)
    if mode == 'trusted' and score >= threshold and metrics['vis_count'] >= 4 and metrics['shoulders_count'] >= 1:
        accept = True

    metrics['human_score'] = score
    return accept, score, metrics


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 2: MEMÓRIA PERSISTENTE# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 2: MEMÓRIA PERSISTENTE                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PersistentMemory:
    def __init__(self, role: str, config: TitanConfig):
        self.role = role
        self.config = config
        self.positions: deque = deque(maxlen=60)
        self.velocity: np.ndarray = np.zeros(2)
        self.last_position: Optional[np.ndarray] = None
        self.last_seen_frame: int = -999
        self.frames_visible: int = 0
        self.track_ids: set = set()
        self.appearance_descriptors: deque = deque(maxlen=config.appearance_history_size)
        self.torso_heights: deque = deque(maxlen=100)
        self.shoulder_widths: deque = deque(maxlen=100)
        self.bbox_heights: deque = deque(maxlen=100)
        self.body_areas: deque = deque(maxlen=100)
        self.last_kpts: Optional[np.ndarray] = None

    def update(self, kpts: np.ndarray, track_id: int, frame_id: int, frame_bgr: np.ndarray):
        if kpts[0][2] > 0.10:
            pos = kpts[0][:2].copy()
        elif kpts[5][2] > 0.10 and kpts[6][2] > 0.10:
            pos = ((kpts[5][:2] + kpts[6][:2]) / 2.0).copy()
        else:
            visible = kpts[kpts[:, 2] > 0.05]
            pos = np.mean(visible[:, :2], axis=0).copy() if len(visible) else kpts[0][:2].copy()

        if self.last_position is not None:
            dt = frame_id - self.last_seen_frame
            if 0 < dt < 30:
                instant_vel = (pos - self.last_position) / dt
                self.velocity = 0.3 * instant_vel + 0.7 * self.velocity

        self.positions.append(pos)
        self.last_position = pos.copy()
        self.last_seen_frame = frame_id
        self.frames_visible += 1
        if track_id >= 0:
            self.track_ids.add(track_id)
        self.last_kpts = kpts.copy()

        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        if all(k[2] > 0.15 for k in [l_sh, r_sh, l_hip, r_hip]):
            mid_sh = (l_sh[:2] + r_sh[:2]) / 2
            mid_hip = (l_hip[:2] + r_hip[:2]) / 2
            self.torso_heights.append(np.linalg.norm(mid_hip - mid_sh))
            self.shoulder_widths.append(np.linalg.norm(r_sh[:2] - l_sh[:2]))
            xs = [l_sh[0], r_sh[0], l_hip[0], r_hip[0]]
            ys = [l_sh[1], r_sh[1], l_hip[1], r_hip[1]]
            self.body_areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))

        vis_y = [kpts[i][1] for i in range(17) if kpts[i][2] > 0.1]
        if len(vis_y) >= 3:
            self.bbox_heights.append(max(vis_y) - min(vis_y))

        interval = getattr(self.config, 'appearance_update_interval', 5)
        if self.frames_visible % interval == 0:
            desc = self._compute_appearance(frame_bgr, kpts)
            if desc is not None:
                self.appearance_descriptors.append(desc)

    def predict_position(self, current_frame: int) -> Optional[np.ndarray]:
        if self.last_position is None:
            return None
        dt = current_frame - self.last_seen_frame
        if dt > self.config.max_frames_predicao:
            return None
        decay = 0.92 ** dt
        return self.last_position + self.velocity * dt * decay

    def is_alive(self, current_frame: int) -> bool:
        if self.last_seen_frame < 0:
            return False
        return (current_frame - self.last_seen_frame) <= self.config.memory_max_frames

    def frames_missing(self, current_frame: int) -> int:
        return current_frame - self.last_seen_frame

    def get_median_size(self) -> float:
        h = np.median(self.torso_heights) if self.torso_heights else 0
        w = np.median(self.shoulder_widths) if self.shoulder_widths else 0
        a = np.sqrt(np.median(self.body_areas)) if self.body_areas else 0
        cfg = self.config
        return cfg.peso_altura_tronco * h + cfg.peso_largura_ombros * w + cfg.peso_bbox_area * a

    def get_mean_appearance(self) -> Optional[np.ndarray]:
        if not self.appearance_descriptors:
            return None
        return np.mean(self.appearance_descriptors, axis=0)

    def compare_appearance(self, descriptor: np.ndarray) -> float:
        mean = self.get_mean_appearance()
        if mean is None:
            return 0.0
        corr = float(cv2.compareHist(
            mean.astype(np.float32).reshape(-1, 1),
            descriptor.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        ))
        return max(0.0, min(1.0, (corr + 1.0) / 2.0))

    def compare_position(self, pos: np.ndarray, current_frame: int) -> float:
        predicted = self.predict_position(current_frame)
        if predicted is None:
            return 0.0
        dist = np.linalg.norm(pos - predicted)
        return float(np.exp(-dist / self.config.position_norm_sigma))

    def compare_size(self, kpts: np.ndarray) -> float:
        if not self.torso_heights:
            return 0.0
        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        if not all(k[2] > 0.15 for k in [l_sh, r_sh, l_hip, r_hip]):
            return 0.5
        mid_sh = (l_sh[:2] + r_sh[:2]) / 2
        mid_hip = (l_hip[:2] + r_hip[:2]) / 2
        h = np.linalg.norm(mid_hip - mid_sh)
        median_h = np.median(self.torso_heights)
        if median_h < 1:
            return 0.5
        ratio = min(h, median_h) / max(h, median_h)
        return float(ratio)

    def _compute_appearance(self, frame: np.ndarray, kpts: np.ndarray) -> Optional[np.ndarray]:
        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        if not all(k[2] > 0.15 for k in [l_sh, r_sh, l_hip, r_hip]):
            return None
        xs = [l_sh[0], r_sh[0], l_hip[0], r_hip[0]]
        ys = [l_sh[1], r_sh[1], l_hip[1], r_hip[1]]
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, int(min(xs)) - 10)
        y1 = max(0, int(min(ys)) - 10)
        x2 = min(w_frame, int(max(xs)) + 10)
        y2 = min(h_frame, int(max(ys)) + 10)
        if x2 - x1 < 15 or y2 - y1 < 15:
            return None
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bins = self.config.hist_bins
        hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 3: GAZE ESTIMATION v6 — MEDIAPIPE + IRIS LANDMARKS               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class GazeEstimator:
    """
    Estimativa de atenção v6 — MediaPipe Face Mesh + Íris em 2D puro.

    MUDANÇA v6 vs v5:
    v5 usava apenas direção da face (nose_tip - face_center) para gaze.
    v6 adiciona landmarks da ÍRIS (468–477, disponíveis com refine_landmarks=True):
      - left iris center = landmark 468
      - right iris center = landmark 473
    
    A íris dentro do socket ocular dá a direção do OLHAR de verdade.
    Um rosto virado 30° mas com olhos olhando para o lado oposto
    agora é detectado corretamente.
    
    PESO DINÂMICO POR FRONTALIDADE:
    - Face frontal (frontality=1.0): íris 55% + face 45%
    - Face de perfil (frontality=0.0): face 100% (íris ocluída = não confiável)
    
    Frontality é estimada pela posição do nariz relativa às têmporas.
    """

    FACE_CONTOUR_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    LEFT_EYE_INDICES  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    NOSE_INDICES  = [1, 2, 98, 327]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    # Face direction landmarks
    NOSE_TIP    = 1
    NOSE_BRIDGE = 168
    FOREHEAD    = 10
    CHIN        = 152
    LEFT_TEMPLE  = 234
    RIGHT_TEMPLE = 454
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER  = 133   # nasal corner left eye
    RIGHT_EYE_INNER = 362   # nasal corner right eye

    # ★ v6: Iris landmarks (require refine_landmarks=True) ★
    LEFT_IRIS_CENTER  = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_IRIS_PERIMETER  = [469, 470, 471, 472]
    RIGHT_IRIS_PERIMETER = [474, 475, 476, 477]

    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._mp_available = MEDIAPIPE_AVAILABLE
        self._face_mesh = None
        self._mp_initialized = False
        self._mp_fail_count = 0
        self._last_mp_results = {}

    def _init_mediapipe(self):
        if self._mp_initialized:
            return
        self._mp_initialized = True
        if not MEDIAPIPE_AVAILABLE:
            self._mp_available = False
            return
        try:
            _se = sys.stderr
            _devnull_mp = None
            try:
                _devnull_mp = open(os.devnull, 'w')
                sys.stderr = _devnull_mp
            except Exception:
                pass
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,  # OBRIGATÓRIO para íris (landmarks 468-477)
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
            )
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._face_mesh.process(dummy)
            sys.stderr = _se if _se and not _se.closed else sys.__stderr__
            if _devnull_mp is not None:
                try:
                    _devnull_mp.close()
                except Exception:
                    pass
            print("    ✓ MediaPipe Face Mesh v6 lazy-init OK (íris 468-477 ativos)", flush=True)
        except Exception as e:
            sys.stderr = _se if _se and not _se.closed else sys.__stderr__
            if _devnull_mp is not None:
                try:
                    _devnull_mp.close()
                except Exception:
                    pass
            print(f"    ✗ MediaPipe init falhou: {e}", flush=True)
            self._mp_available = False
            self._face_mesh = None

    def _crop_head_region(self, frame, kpts):
        h, w = frame.shape[:2]
        head_pts = []
        for idx in [0, 1, 2, 3, 4]:
            if kpts[idx][2] > 0.04:
                head_pts.append(kpts[idx][:2])

        mid_sh = None
        shoulder_width = None
        if kpts[5][2] > 0.06 and kpts[6][2] > 0.06:
            mid_sh = (kpts[5][:2] + kpts[6][:2]) / 2.0
            shoulder_width = np.linalg.norm(kpts[6][:2] - kpts[5][:2])

        if not head_pts:
            if mid_sh is None or shoulder_width is None:
                return None, 0, 0
            head_pts.append(mid_sh - np.array([0.0, shoulder_width * 0.90], dtype=np.float32))

        head_pts = np.array(head_pts, dtype=np.float32)
        cx, cy = np.mean(head_pts, axis=0)
        spread = float(np.max(np.ptp(head_pts, axis=0))) if len(head_pts) >= 2 else 0.0

        crop_size = max(120.0, spread * 4.0)
        if shoulder_width is not None:
            crop_size = max(crop_size, shoulder_width * 2.3)
        crop_size = min(crop_size, max(h, w) * 0.45)

        if mid_sh is not None and shoulder_width is not None:
            cy = min(cy, float(mid_sh[1] - shoulder_width * 0.30))

        half = int(crop_size / 2.0)
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))

        if mid_sh is not None and shoulder_width is not None:
            lower_cap = int(min(h, mid_sh[1] + shoulder_width * 0.55))
            y2 = min(y2, lower_cap)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            return None, 0, 0
        return frame[y1:y2, x1:x2], x1, y1


    def _get_face_direction_2d(self, crop, ox, oy):
        """
        Roda MediaPipe e retorna direção da face em coordenadas da imagem completa.

        v54: crop mais contido + confiança dinâmica baseada em frontalidade,
        tamanho útil do rosto e contribuição real da íris.
        """
        try:
            if crop is None or crop.size == 0:
                return None, None, 0.0
            ch, cw = crop.shape[:2]
            if ch < 30 or cw < 30:
                return None, None, 0.0

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None, None, 0.0

            face = results.multi_face_landmarks[0]
            landmarks_px = []
            for lm in face.landmark:
                px = lm.x * cw + ox
                py = lm.y * ch + oy
                landmarks_px.append((px, py))

            nose_tip = np.array(landmarks_px[self.NOSE_TIP], dtype=np.float32)
            nose_bridge = np.array(landmarks_px[self.NOSE_BRIDGE], dtype=np.float32)
            forehead = np.array(landmarks_px[self.FOREHEAD], dtype=np.float32)
            chin = np.array(landmarks_px[self.CHIN], dtype=np.float32)
            face_center = (nose_bridge + forehead + chin) / 3.0

            v1 = nose_tip - face_center
            n1 = np.linalg.norm(v1)

            l_temple = np.array(landmarks_px[self.LEFT_TEMPLE], dtype=np.float32)
            r_temple = np.array(landmarks_px[self.RIGHT_TEMPLE], dtype=np.float32)
            temple_vec = r_temple - l_temple
            face_normal = np.array([-temple_vec[1], temple_vec[0]], dtype=np.float32)
            n2 = np.linalg.norm(face_normal)

            if n2 > 1e-6:
                face_normal /= n2
                mid_temple = (l_temple + r_temple) / 2.0
                if np.dot(face_normal, nose_tip - mid_temple) < 0:
                    face_normal = -face_normal

            if n1 > 1e-6 and n2 > 1e-6:
                v1_norm = v1 / n1
                face_gaze = v1_norm * 0.6 + face_normal * 0.4
            elif n1 > 1e-6:
                face_gaze = v1 / n1
            elif n2 > 1e-6:
                face_gaze = face_normal
            else:
                return None, landmarks_px, 0.0

            fg_n = np.linalg.norm(face_gaze)
            if fg_n < 1e-6:
                return None, landmarks_px, 0.0
            face_gaze /= fg_n

            iris_gaze = None
            iris_weight = 0.0
            frontality = 0.0
            iris_strength = 0.0

            if len(landmarks_px) >= 478:
                try:
                    left_eye_pts = [np.array(landmarks_px[i], dtype=np.float32) for i in self.LEFT_EYE_INDICES if i < len(landmarks_px)]
                    right_eye_pts = [np.array(landmarks_px[i], dtype=np.float32) for i in self.RIGHT_EYE_INDICES if i < len(landmarks_px)]
                    left_iris_pt = np.array(landmarks_px[self.LEFT_IRIS_CENTER], dtype=np.float32)
                    right_iris_pt = np.array(landmarks_px[self.RIGHT_IRIS_CENTER], dtype=np.float32)

                    if left_eye_pts and right_eye_pts:
                        left_eye_center = np.mean(left_eye_pts, axis=0)
                        right_eye_center = np.mean(right_eye_pts, axis=0)
                        left_eye_width = max(np.linalg.norm(np.array(landmarks_px[self.LEFT_EYE_OUTER]) - np.array(landmarks_px[self.LEFT_EYE_INNER])), 1.0)
                        right_eye_width = max(np.linalg.norm(np.array(landmarks_px[self.RIGHT_EYE_OUTER]) - np.array(landmarks_px[self.RIGHT_EYE_INNER])), 1.0)

                        left_offset = (left_iris_pt - left_eye_center) / left_eye_width
                        right_offset = (right_iris_pt - right_eye_center) / right_eye_width
                        iris_dir_raw = (left_offset + right_offset) / 2.0
                        iris_n = np.linalg.norm(iris_dir_raw)

                        temple_span = max(np.linalg.norm(r_temple - l_temple), 1.0)
                        mid_temple_x = (l_temple[0] + r_temple[0]) / 2.0
                        lateral_offset_ratio = abs(nose_tip[0] - mid_temple_x) / max(temple_span / 2.0, 1.0)
                        frontality = max(0.0, 1.0 - lateral_offset_ratio)

                        if iris_n > 0.015:
                            iris_gaze = iris_dir_raw / iris_n
                            iris_strength = float(min(1.0, iris_n / 0.10))
                            iris_weight = frontality * iris_strength * 0.55
                except Exception:
                    iris_gaze = None
                    iris_weight = 0.0
                    frontality = 0.0
                    iris_strength = 0.0

            if iris_gaze is not None and iris_weight > 0.05:
                face_weight = 1.0 - iris_weight
                combined = face_gaze * face_weight + iris_gaze * iris_weight
                cn = np.linalg.norm(combined)
                gaze = combined / cn if cn > 1e-6 else face_gaze
            else:
                gaze = face_gaze

            face_scale = min(min(ch, cw) / 180.0, 1.0)
            confidence = 0.35 + 0.25 * frontality + 0.15 * face_scale + 0.15 * iris_weight / 0.55 + 0.10 * iris_strength
            confidence = float(max(0.25, min(0.95, confidence)))
            return gaze, landmarks_px, confidence
        except Exception:
            self._mp_fail_count += 1
            if self._mp_fail_count > 100:
                self._mp_available = False
            return None, None, 0.0


    def estimate_looking(self, kpts, target_pos, frame=None, role=None,
                         mediapipe_min_yolo_conf=0.0, camera_y_weight=0.3):
        obs_center = self._best_center(kpts)
        vec_to_target_raw = target_pos - obs_center
        dist = np.linalg.norm(vec_to_target_raw)
        if dist < 1e-6:
            return 0.0, 0.0
        vec_to_target_unit = vec_to_target_raw / dist

        vec_attenuated = np.array([vec_to_target_raw[0], vec_to_target_raw[1] * camera_y_weight])
        att_dist = np.linalg.norm(vec_attenuated)
        vec_att_norm = vec_to_target_unit if att_dist < 1e-6 else (vec_attenuated / att_dist)

        mp_score = None
        mp_conf = 0.0
        mp_landmarks = None
        mp_gaze = None

        if frame is not None and self._mp_available:
            if not self._mp_initialized:
                self._init_mediapipe()
            if self._mp_available and self._face_mesh is not None:
                try:
                    crop, ox, oy = self._crop_head_region(frame, kpts)
                    if crop is not None:
                        gaze_2d, landmarks_px, conf = self._get_face_direction_2d(crop, ox, oy)
                        if gaze_2d is not None:
                            gaze_att = np.array([gaze_2d[0], gaze_2d[1] * camera_y_weight])
                            gn = np.linalg.norm(gaze_att)
                            if gn > 1e-6:
                                gaze_att /= gn
                            mp_score = float(np.dot(gaze_att, vec_att_norm))
                            mp_conf = conf
                            mp_landmarks = landmarks_px
                            mp_gaze = gaze_2d
                except Exception:
                    pass

        yolo_score, yolo_conf = self._yolo_estimate(kpts, vec_to_target_unit, camera_y_weight)

        mp_min_conf = max(0.35, float(mediapipe_min_yolo_conf or 0.0))
        if mp_score is not None and mp_conf >= mp_min_conf:
            if yolo_conf > 0.08:
                final_score = mp_score * 0.78 + yolo_score * 0.22
                final_conf = mp_conf * 0.80 + yolo_conf * 0.20
            else:
                final_score = mp_score
                final_conf = mp_conf
            used_mp = True
        elif mp_score is not None:
            final_score = mp_score * 0.45 + yolo_score * 0.55
            final_conf = max(mp_conf, yolo_conf) * 0.75
            used_mp = True
        else:
            final_score = yolo_score
            final_conf = yolo_conf
            used_mp = False

        if role:
            self._last_mp_results[role] = {
                'landmarks_px': mp_landmarks,
                'gaze_vec': mp_gaze,
                'used_mp': used_mp,
                'score': final_score,
            }

        return final_score, final_conf
    def draw_mediapipe(self, frame, role, color):
        """Desenha face mesh + íris + vetor de gaze."""
        info = self._last_mp_results.get(role)
        if not info or not info.get('landmarks_px'):
            return

        landmarks = info['landmarks_px']
        gaze = info.get('gaze_vec')
        used_mp = info.get('used_mp', False)

        if not used_mp:
            return

        lm_color = (255, 255, 0)

        # Contorno facial
        pts_contour = [landmarks[i] for i in self.FACE_CONTOUR_INDICES if i < len(landmarks)]
        for i in range(len(pts_contour)):
            p1 = (int(pts_contour[i][0]),                          int(pts_contour[i][1]))
            p2 = (int(pts_contour[(i+1) % len(pts_contour)][0]),   int(pts_contour[(i+1) % len(pts_contour)][1]))
            cv2.line(frame, p1, p2, lm_color, 1, cv2.LINE_AA)

        # Olhos
        for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
            pts = [landmarks[i] for i in eye_indices if i < len(landmarks)]
            for i in range(len(pts)):
                p1 = (int(pts[i][0]),              int(pts[i][1]))
                p2 = (int(pts[(i+1)%len(pts)][0]), int(pts[(i+1)%len(pts)][1]))
                cv2.line(frame, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

        # Nariz
        for idx in self.NOSE_INDICES:
            if idx < len(landmarks):
                pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
                cv2.circle(frame, pt, 2, (0, 200, 255), -1)

        nose = landmarks[self.NOSE_TIP]
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 4, (0, 0, 255), -1)

        # ★ v6: Íris landmarks ★
        if len(landmarks) >= 478:
            # Centro da íris (ponto verde)
            for iris_center_idx in [self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER]:
                if iris_center_idx < len(landmarks):
                    pt = (int(landmarks[iris_center_idx][0]), int(landmarks[iris_center_idx][1]))
                    cv2.circle(frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

            # Perímetro da íris (círculo verde fino)
            for perimeter_indices in [self.LEFT_IRIS_PERIMETER, self.RIGHT_IRIS_PERIMETER]:
                pts = [landmarks[i] for i in perimeter_indices if i < len(landmarks)]
                if len(pts) == 4:
                    for i in range(4):
                        p1 = (int(pts[i][0]),          int(pts[i][1]))
                        p2 = (int(pts[(i+1)%4][0]),    int(pts[(i+1)%4][1]))
                        cv2.line(frame, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)

        # Vetor de gaze
        if gaze is not None:
            start = (int(nose[0]), int(nose[1]))
            end   = (int(nose[0] + gaze[0] * 60), int(nose[1] + gaze[1] * 60))
            cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)

        cv2.putText(frame, "MP+IRIS", (int(nose[0]) + 8, int(nose[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    def _yolo_estimate(self, kpts, vec_to_target, camera_y_weight=0.3):
        nose = kpts[0]; l_eye = kpts[1]; r_eye = kpts[2]
        l_ear = kpts[3]; r_ear = kpts[4]
        l_sh = kpts[5]; r_sh = kpts[6]

        def y_att_dot(gaze_vec, target_vec):
            ga = np.array([gaze_vec[0], gaze_vec[1] * camera_y_weight])
            ta = np.array([target_vec[0], target_vec[1] * camera_y_weight])
            gn = np.linalg.norm(ga)
            tn = np.linalg.norm(ta)
            if gn < 1e-6 or tn < 1e-6:
                return 0.0
            return float(np.dot(ga / gn, ta / tn))

        head_scores = []

        if nose[2] > 0.06 and l_eye[2] > 0.06 and r_eye[2] > 0.06:
            mid_eyes = (l_eye[:2] + r_eye[:2]) / 2.0
            gaze = nose[:2] - mid_eyes
            n = np.linalg.norm(gaze)
            if n > 1e-6:
                s = y_att_dot(gaze, vec_to_target)
                avg_c = (nose[2] + l_eye[2] + r_eye[2]) / 3.0
                head_scores.append((s, 0.50 * min(avg_c / 0.5, 1.0)))

        l_ear_vis = l_ear[2] > 0.04
        r_ear_vis = r_ear[2] > 0.04
        if l_ear_vis != r_ear_vis:
            vis_ear = l_ear[:2] if l_ear_vis else r_ear[:2]
            if nose[2] > 0.03:
                hd = nose[:2] - vis_ear
                n = np.linalg.norm(hd)
                if n > 1e-6:
                    head_scores.append((y_att_dot(hd, vec_to_target), 0.40))

        if l_ear_vis and r_ear_vis:
            ev = r_ear[:2] - l_ear[:2]
            fn = np.array([-ev[1], ev[0]])
            n = np.linalg.norm(fn)
            if n > 1e-6:
                fn /= n
                if nose[2] > 0.03:
                    m = (l_ear[:2] + r_ear[:2]) / 2.0
                    if np.dot(fn, nose[:2] - m) < 0:
                        fn = -fn
                head_scores.append((y_att_dot(fn, vec_to_target), 0.35))

        body_score = 0.0
        body_conf  = 0.0
        sh_vis = l_sh[2] > 0.08 and r_sh[2] > 0.08
        if sh_vis:
            mid_sh = (l_sh[:2] + r_sh[:2]) / 2.0
            sv = r_sh[:2] - l_sh[:2]
            bn = np.array([-sv[1], sv[0]])
            n = np.linalg.norm(bn)
            if n > 1e-6:
                bn /= n
                for idx in [0,1,2,3,4]:
                    if kpts[idx][2] > 0.02:
                        if np.dot(bn, kpts[idx][:2] - mid_sh) < 0:
                            bn = -bn
                        break
                body_score = y_att_dot(bn, vec_to_target)
                body_conf  = 0.40

        if head_scores:
            hw = sum(w for _, w in head_scores)
            hs = sum(s * w for s, w in head_scores) / max(hw, 0.01)
            if body_conf > 0:
                head_strength = min(hw / 0.50, 1.0)
                body_weight = 0.15 + (1.0 - head_strength) * 0.35
                head_weight = 1.0 - body_weight
                final = hs * head_weight + body_score * body_weight
                conf  = min(1.0, hw * 0.7 + body_conf * 0.3)
            else:
                final = hs
                conf  = min(1.0, hw * 0.9)
        elif body_conf > 0:
            final = body_score
            conf  = body_conf * 0.70
        else:
            return 0.0, 0.0
        return final, conf

    @staticmethod
    def _best_center(kpts):
        if kpts[0][2] > 0.1:
            return kpts[0][:2]
        if kpts[5][2] > 0.1 and kpts[6][2] > 0.1:
            return (kpts[5][:2] + kpts[6][:2]) / 2.0
        visible = kpts[kpts[:, 2] > 0.05]
        if len(visible) > 0:
            return np.mean(visible[:, :2], axis=0)
        return kpts[0][:2]

    @staticmethod
    def estimate(kpts):
        nose = kpts[0]; l_eye = kpts[1]; r_eye = kpts[2]
        l_ear = kpts[3]; r_ear = kpts[4]; l_sh = kpts[5]; r_sh = kpts[6]
        if nose[2] > 0.05:
            ref = None
            if l_eye[2] > 0.05 and r_eye[2] > 0.05: ref = (l_eye[:2] + r_eye[:2]) / 2.0
            elif l_eye[2] > 0.05: ref = l_eye[:2]
            elif r_eye[2] > 0.05: ref = r_eye[:2]
            elif l_ear[2] > 0.04: ref = l_ear[:2]
            elif r_ear[2] > 0.04: ref = r_ear[:2]
            if ref is not None:
                v = nose[:2] - ref
                n = np.linalg.norm(v)
                if n > 1e-6: return v / n
        if l_ear[2] > 0.04 and r_ear[2] > 0.04:
            ev = r_ear[:2] - l_ear[:2]
            fn = np.array([-ev[1], ev[0]])
            n = np.linalg.norm(fn)
            if n > 1e-6:
                fn /= n
                if nose[2] > 0.03:
                    m = (l_ear[:2] + r_ear[:2]) / 2.0
                    if np.dot(fn, nose[:2] - m) < 0: fn = -fn
                return fn
        if l_sh[2] > 0.08 and r_sh[2] > 0.08:
            sv = r_sh[:2] - l_sh[:2]
            bn = np.array([-sv[1], sv[0]])
            n = np.linalg.norm(bn)
            if n > 1e-6:
                bn /= n
                if nose[2] > 0.03:
                    m = (l_sh[:2] + r_sh[:2]) / 2.0
                    if np.dot(bn, nose[:2] - m) < 0: bn = -bn
                return bn
        return None

    @staticmethod
    def confidence(kpts):
        fv = sum(1 for i in [0,1,2,3,4] if kpts[i][2] > 0.05)
        fc = [kpts[i][2] for i in [0,1,2,3,4] if kpts[i][2] > 0]
        fs = (fv / 5.0) * (np.mean(fc) if fc else 0)
        bv = sum(1 for i in [5,6,11,12] if kpts[i][2] > 0.08)
        bc = [kpts[i][2] for i in [5,6,11,12] if kpts[i][2] > 0]
        bs = (bv / 4.0) * (np.mean(bc) if bc else 0) * 0.5
        return max(fs, bs)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 4: SUAVIZAÇÃO TEMPORAL                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TemporalSmoother:
    def __init__(self, config: TitanConfig):
        self.alpha    = config.ema_alpha
        self.window   = config.voting_window_size
        self.th_enter = config.looking_threshold_enter
        self.th_exit  = config.looking_threshold_exit
        self.ema:   Dict[str, float]  = {}
        self.votes: Dict[str, deque]  = {}
        self.state: Dict[str, bool]   = {}

    def update(self, role: str, raw_score: float) -> Tuple[bool, float]:
        if role not in self.ema:
            self.ema[role]   = raw_score
            self.votes[role] = deque(maxlen=self.window)
            self.state[role] = False

        self.ema[role] = self.alpha * raw_score + (1 - self.alpha) * self.ema[role]
        smoothed = self.ema[role]
        was = self.state[role]
        instant = smoothed > (self.th_exit if was else self.th_enter)
        self.votes[role].append(instant)
        final = sum(self.votes[role]) > len(self.votes[role]) / 2
        self.state[role] = final
        return final, smoothed

    def reset(self, role: Optional[str] = None):
        targets = [role] if role else list(self.ema.keys())
        for r in targets:
            self.ema.pop(r, None)
            self.votes.pop(r, None)
            self.state.pop(r, None)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 5: DETECTOR MULTI-PASS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class MultiPassDetector:
    def __init__(self, model: YOLO, config: TitanConfig, device, use_half: bool):
        self.model = model
        self.config = config
        self.passes = config.detection_passes
        self.device = device
        self.use_half = use_half
        self.imgsz = config.imgsz
        self.use_tta = config.use_tta
        self._consecutive_found = 0
        self._multipass_cooldown = 0
        self.known_ids: set = set()
        self._next_ghost_id: int = -1
        self.last_used_extra_passes = False

    def _next_temp_track_id(self) -> int:
        tid = self._next_ghost_id
        self._next_ghost_id -= 1
        return tid

    def _person_quality(self, human_score: float, box_conf: float, metrics: Dict[str, Any], source: str) -> float:
        q = 0.60 * human_score
        q += 0.15 * min(max(box_conf, 0.0) / 0.60, 1.0)
        q += 0.15 * min(metrics.get('vis_count', 0) / 10.0, 1.0)
        q += 0.10 * min(metrics.get('mean_conf', 0.0) / 0.60, 1.0)
        if source == 'track':
            q += 0.03
        return float(max(0.0, min(1.2, q)))

    def _extract_candidates(self, results, default_mode: str = 'strict', source: str = 'track') -> list:
        people = []
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return people

        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        confs = results[0].boxes.conf.cpu().numpy() if getattr(results[0].boxes, 'conf', None) is not None else np.ones(len(keypoints), dtype=np.float32)
        ids_tensor = getattr(results[0].boxes, 'id', None)
        track_ids = ids_tensor.cpu().numpy().astype(int) if ids_tensor is not None else None

        for i, kpts in enumerate(keypoints):
            box = boxes[i] if i < len(boxes) else None
            det_conf = float(confs[i]) if i < len(confs) else 0.0
            if source == 'track' and det_conf < self.config.min_box_conf_tracked:
                continue
            if source != 'track' and det_conf < self.config.min_box_conf_extra:
                continue

            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else self._next_temp_track_id()
            mode = 'trusted' if (tid >= 0 and tid in self.known_ids) else ('extra' if source != 'track' else default_mode)
            ok, human_score, metrics = score_human_pose(kpts, box, det_conf, mode=mode, config=self.config)
            if not ok:
                continue

            center = metrics['center']
            quality = self._person_quality(human_score, det_conf, metrics, source)
            people.append({
                'kpts': kpts,
                'track_id': tid,
                'center': center,
                'has_tracking': bool(track_ids is not None and i < len(track_ids)),
                'bbox': box,
                'det_conf': det_conf,
                'human_score': human_score,
                'quality': quality,
                'metrics': metrics,
                'source': source,
            })
        people.sort(key=lambda p: (p['quality'], p['human_score'], p['det_conf']), reverse=True)
        return people

    def _merge_people(self, base_people: list, extra_people: list) -> list:
        merged = list(base_people)
        for extra in extra_people:
            duplicate_idx = None
            for idx, existing in enumerate(merged):
                center_dist = np.linalg.norm(extra['center'] - existing['center'])
                iou = bbox_iou(extra.get('bbox'), existing.get('bbox'))
                if center_dist < self.config.duplicate_distance or iou > 0.45:
                    duplicate_idx = idx
                    break

            if duplicate_idx is None:
                merged.append(extra)
                continue

            existing = merged[duplicate_idx]
            replace = False
            if extra['has_tracking'] and not existing['has_tracking']:
                replace = True
            elif extra['quality'] > existing['quality'] + 0.08:
                replace = True
            elif extra['quality'] > existing['quality'] and extra['human_score'] > existing['human_score'] + 0.05:
                replace = True
            if replace:
                if existing['has_tracking'] and not extra['has_tracking']:
                    extra['track_id'] = existing['track_id']
                merged[duplicate_idx] = extra

        merged.sort(key=lambda p: (p['quality'], p['human_score'], p['det_conf']), reverse=True)
        return merged[:self.config.candidate_pool_max]

    def detect(self, frame: np.ndarray, prefer_recovery: bool = False) -> Tuple[list, Any]:
        p1 = self.passes[0]
        results = self.model.track(
            frame, persist=True, verbose=False,
            tracker="botsort.yaml", classes=[0],
            conf=p1["conf"], iou=p1["iou"],
            half=self.use_half, imgsz=self.imgsz, device=self.device,
        )

        people = self._extract_candidates(results, default_mode='strict', source='track')
        if len(people) >= 2:
            self._consecutive_found += 1
            self._multipass_cooldown = 20
            self.last_used_extra_passes = False
            return people, results

        need_extra = prefer_recovery or self._multipass_cooldown <= 0 or len(people) < 2
        if not need_extra and self._multipass_cooldown > 0:
            self._multipass_cooldown -= 1
            self._consecutive_found = 0
            self.last_used_extra_passes = False
            return people, results

        used_extra = False
        merged = list(people)
        for pass_cfg in self.passes[1:]:
            extra_results = self.model.predict(
                frame, verbose=False, classes=[0],
                conf=pass_cfg["conf"], iou=pass_cfg["iou"],
                half=self.use_half, imgsz=self.imgsz,
                augment=self.use_tta, device=self.device,
            )
            extra_people = self._extract_candidates(extra_results, default_mode='extra', source='extra')
            merged = self._merge_people(merged, extra_people)
            used_extra = True
            if len(merged) >= 2:
                break

        self.last_used_extra_passes = used_extra
        if len(merged) >= 2:
            self._consecutive_found = 1
            self._multipass_cooldown = 20
        elif self._multipass_cooldown > 0:
            self._multipass_cooldown -= 1
        return merged, results

    def _extract_valid(self, results) -> list:
        return self._extract_candidates(results, default_mode='strict', source='track')

    def _extract_valid_no_track(self, results) -> list:
        return self._extract_candidates(results, default_mode='extra', source='extra')

    def _is_valid_human(self, kpts: np.ndarray, box: np.ndarray = None,
                        mode: str = 'strict') -> bool:
        ok, _, _ = score_human_pose(kpts, box, None, mode=mode, config=self.config)
        return ok

    def _is_valid(self, kpts: np.ndarray) -> bool:
        return self._is_valid_human(kpts)

    def _get_center(self, kpts: np.ndarray) -> np.ndarray:
        return body_metrics_from_pose(kpts)['center']

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 6: IDENTITY MANAGER# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 6: IDENTITY MANAGER                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# Maxlen para a janela deslizante de acumulação
_ACCUM_MAXLEN = 90
_ACCUM_APP_MAXLEN = 30

class IdentityManager:
    """
    v58: gerenciamento de papéis orientado a PAR, com anti-ping-pong.

    Ideias centrais:
    1) papel é decidido pelo PAR inteiro, não por score isolado;
    2) bans temporários por (track_id, papel) evitam reacoplar o mesmo erro;
    3) janelas de incerteza suspendem gravação ao detectar instabilidade;
    4) reset por inversão persistente prefere NO DATA a trocar papel errado;
    5) histórico dos tracks permanece vivo para relock rápido e robusto.

    v58 additions:
    6) lock threshold adaptativo pra pares de tamanho similar (ratio < 1.35);
    7) honeymoon ignora bad_rate puro em similar_pair_mode (só swap_rate conta);
    8) floor penalties escalam proporcionalmente ao prescan ratio;
    9) sw_reliable flag reduz peso de shoulder_width quando prescan inconsistente;
    10) re-lock acelerado (amostras/2) quando memória persistente está viva;
    11) anti-cascata: grace period de 30f pra pair-ambiguous cooldown.
    """

    def __init__(self, config: TitanConfig):
        self.config = config
        self.memories: Dict[str, PersistentMemory] = {
            'GUARDIAN': PersistentMemory('GUARDIAN', config),
            'CHILD': PersistentMemory('CHILD', config),
        }
        self.identities: Dict[int, str] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.is_locked: bool = False

        self.feature_history: Dict[int, Dict[str, deque]] = defaultdict(
            lambda: {
                'torso_height': deque(maxlen=_ACCUM_MAXLEN),
                'shoulder_width': deque(maxlen=_ACCUM_MAXLEN),
                'body_area': deque(maxlen=_ACCUM_MAXLEN),
                'bbox_height': deque(maxlen=_ACCUM_MAXLEN),
                'center_y': deque(maxlen=_ACCUM_MAXLEN),
                'size_signature': deque(maxlen=_ACCUM_MAXLEN),
                'human_score': deque(maxlen=_ACCUM_MAXLEN),
                'quality': deque(maxlen=_ACCUM_MAXLEN),
                'det_conf': deque(maxlen=_ACCUM_MAXLEN),
                'appearance': deque(maxlen=_ACCUM_APP_MAXLEN),
            }
        )

        self.track_observation_count: Dict[int, int] = defaultdict(int)
        self.track_good_count: Dict[int, int] = defaultdict(int)
        self.track_quality_ema: Dict[int, float] = {}
        self.role_ban_until: Dict[Tuple[int, str], int] = {}
        self.assignment_cooldown_until: int = -1
        self.last_instability_reason: str = ''
        self.lock_anchor_profiles: Optional[Dict[str, Dict[str, float]]] = None

        self.lock_frame: int = -1
        self.prescan_profiles: Optional[dict] = None
        self.role_mismatch_streak: Dict[str, int] = {'GUARDIAN': 0, 'CHILD': 0}
        self.INTRUDER_CONFIRM_FRAMES: int = 15
        self.inversion_streak: int = 0
        self.INVERSION_SWAP_FRAMES: int = 12

        self.honeymoon_active: bool = False
        self.honeymoon_frame_count: int = 0
        self.honeymoon_swap_votes: int = 0
        self.honeymoon_bad_quality_votes: int = 0
        self.HONEYMOON_FRAMES: int = 90
        self.HONEYMOON_SWAP_THRESHOLD: float = 0.30

        # ── v58: adaptive thresholds for similar-size pairs ──
        self.adaptive_lock_pair_min_score: float = config.lock_pair_min_score
        self.adaptive_lock_pair_min_margin: float = config.lock_pair_min_margin
        self.similar_pair_mode: bool = False
        self.pair_ambiguous_grace_until: int = -1  # v58: anti-cascade cooldown

        self.event_logger = None

    def _emit_identity_event(self, msg: str):
        if self.event_logger is not None:
            self.event_logger(msg)
        elif self.config.console_identity_events:
            print(msg, flush=True)

    def in_uncertainty_cooldown(self, current_frame: int) -> bool:
        return current_frame < self.assignment_cooldown_until

    def can_attempt_lock(self, current_frame: int) -> bool:
        return (not self.is_locked) and (not self.in_uncertainty_cooldown(current_frame))

    def enter_uncertainty_cooldown(self, current_frame: int, reason: str, frames: Optional[int] = None):
        frames = self.config.role_uncertainty_cooldown_frames if frames is None else max(0, int(frames))
        self.assignment_cooldown_until = max(self.assignment_cooldown_until, current_frame + frames)
        self.last_instability_reason = reason or self.last_instability_reason
        if frames > 0:
            self._emit_identity_event(
                f"[Frame {current_frame}] NO DATA cooldown ({frames}f): {self.last_instability_reason}"
            )

    def register_visible_tracks(self, people: list, frame_id: int):
        for p in people:
            tid = int(p['track_id'])
            if tid < 0:
                continue
            self.track_last_seen[tid] = frame_id
            self.track_observation_count[tid] += 1
            if float(p.get('human_score', 0.0)) >= self.config.human_score_trusted:
                self.track_good_count[tid] += 1
            q = float(p.get('quality', p.get('human_score', 0.0)))
            prev = self.track_quality_ema.get(tid, q)
            self.track_quality_ema[tid] = 0.78 * prev + 0.22 * q

    def get_role(self, track_id: int) -> Optional[str]:
        return self.identities.get(track_id)

    def set_role_track(self, role: str, track_id: int):
        for tid, mapped_role in list(self.identities.items()):
            if mapped_role == role or tid == track_id:
                del self.identities[tid]
        self.identities[int(track_id)] = role
        if int(track_id) >= 0 and int(track_id) not in self.track_last_seen:
            self.track_last_seen[int(track_id)] = self.lock_frame if self.lock_frame >= 0 else 0

    def is_role_track_banned(self, role: str, track_id: int, current_frame: int) -> bool:
        if track_id < 0:
            return False
        return current_frame < self.role_ban_until.get((int(track_id), role), -1)

    def ban_role_track(self, role: str, track_id: int, current_frame: int, reason: str = '', extra_frames: int = 0):
        if track_id < 0:
            return
        until = current_frame + self.config.role_ban_frames + max(0, int(extra_frames))
        key = (int(track_id), role)
        prev = self.role_ban_until.get(key, -1)
        self.role_ban_until[key] = max(prev, until)
        if self.identities.get(int(track_id)) == role:
            self.identities.pop(int(track_id), None)
        why = f" | {reason}" if reason else ''
        self._emit_identity_event(
            f"[Frame {current_frame}] BAN {role} on tid={track_id} until {self.role_ban_until[key]}{why}"
        )

    def reject_role_candidate(self, role: str, person: Optional[dict], current_frame: int, reason: str = ''):
        if person is not None:
            self.ban_role_track(role, int(person['track_id']), current_frame, reason=reason)
        self.role_mismatch_streak[role] = 0
        self.enter_uncertainty_cooldown(
            current_frame,
            reason or f'{role} rejected',
            frames=max(8, self.config.role_uncertainty_cooldown_frames // 2),
        )

    def reset_after_instability(
        self,
        current_frame: int,
        reason: str,
        identified: Optional[Dict[str, dict]] = None,
        ban_current_roles: bool = True,
        extra_ban_frames: int = 0,
    ):
        if ban_current_roles and identified:
            for role, person in identified.items():
                self.ban_role_track(
                    role,
                    int(person['track_id']),
                    current_frame,
                    reason=reason,
                    extra_frames=extra_ban_frames,
                )
        self.soft_reset(clear_histories=False, clear_track_stats=False)
        self.enter_uncertainty_cooldown(current_frame, reason, frames=self.config.role_uncertainty_cooldown_frames)

    def cleanup_stale_ids(self, active_tids: set, current_frame: int):
        stale_positive = []
        for tid, last_seen in list(self.track_last_seen.items()):
            if tid in active_tids:
                continue
            if (current_frame - last_seen) > self.config.identity_stale_frames:
                stale_positive.append(tid)

        for tid in stale_positive:
            self.identities.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.feature_history.pop(tid, None)
            self.track_observation_count.pop(tid, None)
            self.track_good_count.pop(tid, None)
            self.track_quality_ema.pop(tid, None)
            for key in list(self.role_ban_until.keys()):
                if key[0] == tid:
                    self.role_ban_until.pop(key, None)

        for tid in list(self.identities.keys()):
            if tid < 0 and tid not in active_tids:
                self.identities.pop(tid, None)

        for key, until in list(self.role_ban_until.items()):
            if until <= current_frame:
                self.role_ban_until.pop(key, None)

    def _measure_shoulder_width(self, kpts: np.ndarray) -> Optional[float]:
        metrics = body_metrics_from_pose(kpts)
        return metrics.get('shoulder_width')

    def _get_metrics(self, person: Any, bbox: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if isinstance(person, dict):
            metrics = person.get('metrics')
            if metrics is None:
                metrics = body_metrics_from_pose(person['kpts'], person.get('bbox'))
                metrics['human_score'] = person.get('human_score', metrics.get('human_score', 0.0))
            return metrics
        metrics = body_metrics_from_pose(person, bbox)
        metrics.setdefault('human_score', 0.0)
        return metrics

    def _compute_appearance_descriptor(self, frame: Optional[np.ndarray], person: dict) -> Optional[np.ndarray]:
        if frame is None:
            return None
        return self.memories['GUARDIAN']._compute_appearance(frame, person['kpts'])

    def _role_refs(self, role: str) -> Dict[str, Optional[float]]:
        if not self.prescan_profiles:
            return {}
        prefix = role.lower()
        return {
            'sw': self.prescan_profiles.get(f'{prefix}_sw'),
            'bh': self.prescan_profiles.get(f'{prefix}_bh'),
            'torso': self.prescan_profiles.get(f'{prefix}_torso'),
            'size': self.prescan_profiles.get(f'{prefix}_size'),
        }

    def _anchor_refs(self, role: str) -> Dict[str, Optional[float]]:
        if not self.lock_anchor_profiles:
            return {}
        anchor = self.lock_anchor_profiles.get(role, {})
        return {
            'sw': anchor.get('shoulder_width'),
            'bh': anchor.get('bbox_height'),
            'torso': anchor.get('torso_height'),
            'size': anchor.get('size_signature'),
        }

    def _role_absolute_floor(self, role: str) -> Tuple[float, float, float]:
        refs = self._role_refs(role)
        if role == 'GUARDIAN':
            floor_bh = max(120.0, (refs.get('bh') or 0.0) * self.config.guardian_min_height_ratio_to_prescan if refs.get('bh') else 120.0)
            floor_size = max(88.0, (refs.get('size') or 0.0) * self.config.guardian_min_size_ratio_to_prescan if refs.get('size') else 88.0)
            floor_torso = max(48.0, (refs.get('torso') or 0.0) * 0.50 if refs.get('torso') else 48.0)
        else:
            floor_bh = max(self.config.hard_min_bbox_height_if_small, (refs.get('bh') or 0.0) * self.config.child_min_height_ratio_to_prescan if refs.get('bh') else self.config.hard_min_bbox_height_if_small)
            floor_size = max(self.config.hard_min_size_signature_if_small, (refs.get('size') or 0.0) * self.config.child_min_size_ratio_to_prescan if refs.get('size') else self.config.hard_min_size_signature_if_small)
            floor_torso = max(34.0, (refs.get('torso') or 0.0) * 0.45 if refs.get('torso') else 34.0)
        return float(floor_bh), float(floor_size), float(floor_torso)

    def _role_floor_penalty(self, metrics: Dict[str, Any], role: str) -> float:
        bh = float(metrics.get('bbox_height') or 0.0)
        size = float(metrics.get('size_signature') or 0.0)
        torso = float(metrics.get('torso_height') or 0.0) if metrics.get('torso_height') is not None else None
        floor_bh, floor_size, floor_torso = self._role_absolute_floor(role)

        penalty = 0.0
        if bh < floor_bh:
            shortfall = 1.0 - (bh / max(floor_bh, 1.0))
            penalty += 0.18 + 0.32 * max(0.0, shortfall)
        if size < floor_size:
            shortfall = 1.0 - (size / max(floor_size, 1.0))
            penalty += 0.18 + 0.32 * max(0.0, shortfall)
        if torso is not None and torso < floor_torso:
            shortfall = 1.0 - (torso / max(floor_torso, 1.0))
            penalty += 0.08 + 0.18 * max(0.0, shortfall)

        return float(max(0.0, penalty))

    def _pair_target_ratios(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        target_size_ratio = None
        target_bh_ratio = None
        target_sw_ratio = None
        if self.prescan_profiles is not None:
            target_size_ratio = safe_ratio(self.prescan_profiles.get('guardian_size'), self.prescan_profiles.get('child_size'))
            target_bh_ratio = safe_ratio(self.prescan_profiles.get('guardian_bh'), self.prescan_profiles.get('child_bh'))
            target_sw_ratio = safe_ratio(self.prescan_profiles.get('guardian_sw'), self.prescan_profiles.get('child_sw'))
        return target_size_ratio, target_bh_ratio, target_sw_ratio

    def _pair_structural_ambiguity(self, guardian_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> bool:
        actual_size_ratio = safe_ratio(guardian_metrics.get('size_signature'), child_metrics.get('size_signature'))
        actual_bh_ratio = safe_ratio(guardian_metrics.get('bbox_height'), child_metrics.get('bbox_height'))
        actual_sw_ratio = safe_ratio(guardian_metrics.get('shoulder_width'), child_metrics.get('shoulder_width'))
        target_size_ratio, target_bh_ratio, target_sw_ratio = self._pair_target_ratios()

        # ── v58: scale floors by prescan ratio ──
        prescan_ratio = self.prescan_profiles.get('size_ratio', 1.5) if self.prescan_profiles else 1.5
        ratio_factor = max(0.65, min(1.0, (prescan_ratio - 1.0) / 0.60))

        weak_flags = 0
        if actual_size_ratio is not None:
            size_floor = max(1.02, (target_size_ratio or 1.30) * self.config.pair_min_size_ratio_to_prescan * ratio_factor)
            if actual_size_ratio < size_floor:
                weak_flags += 1
        if actual_bh_ratio is not None:
            bh_floor = max(1.02, (target_bh_ratio or 1.25) * self.config.pair_min_bh_ratio_to_prescan * ratio_factor)
            if actual_bh_ratio < bh_floor:
                weak_flags += 1
        if actual_sw_ratio is not None:
            sw_floor = max(1.00, (target_sw_ratio or 1.15) * 0.72 * ratio_factor)
            if actual_sw_ratio < sw_floor:
                weak_flags += 1

        return weak_flags >= 2

    def _pair_ratio_penalty(self, guardian_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> float:
        penalty = 0.0
        actual_size_ratio = safe_ratio(guardian_metrics.get('size_signature'), child_metrics.get('size_signature'))
        actual_bh_ratio = safe_ratio(guardian_metrics.get('bbox_height'), child_metrics.get('bbox_height'))
        target_size_ratio, target_bh_ratio, _ = self._pair_target_ratios()

        # ── v58: scale floors proportionally to prescan ratio ──
        prescan_ratio = self.prescan_profiles.get('size_ratio', 1.5) if self.prescan_profiles else 1.5
        ratio_factor = max(0.65, min(1.0, (prescan_ratio - 1.0) / 0.60))

        if actual_size_ratio is not None:
            size_floor = max(1.02, (target_size_ratio or 1.30) * self.config.pair_min_size_ratio_to_prescan * ratio_factor)
            if actual_size_ratio < size_floor:
                shortfall = (size_floor - actual_size_ratio) / max(size_floor, 1e-6)
                penalty += 0.12 + 0.28 * max(0.0, shortfall)

        if actual_bh_ratio is not None:
            bh_floor = max(1.02, (target_bh_ratio or 1.25) * self.config.pair_min_bh_ratio_to_prescan * ratio_factor)
            if actual_bh_ratio < bh_floor:
                shortfall = (bh_floor - actual_bh_ratio) / max(bh_floor, 1e-6)
                penalty += 0.10 + 0.24 * max(0.0, shortfall)

        if self._pair_structural_ambiguity(guardian_metrics, child_metrics):
            penalty += 0.16

        # Se o CHILD fica abaixo do piso anatômico esperado, penaliza forte o par.
        penalty += min(0.60, self._role_floor_penalty(child_metrics, 'CHILD'))
        penalty += 0.35 * min(0.60, self._role_floor_penalty(guardian_metrics, 'GUARDIAN'))

        return float(max(0.0, penalty))

    def current_pair_structurally_ambiguous(self, identified: Dict[str, dict]) -> bool:
        if 'GUARDIAN' not in identified or 'CHILD' not in identified:
            return True
        return self._pair_structural_ambiguity(
            self._get_metrics(identified['GUARDIAN']),
            self._get_metrics(identified['CHILD']),
        )

    def _track_trust_score(self, track_id: int) -> float:
        if track_id < 0:
            return 0.12
        obs = self.track_observation_count.get(track_id, 0)
        if obs <= 0:
            return 0.0
        good = self.track_good_count.get(track_id, 0)
        good_ratio = good / max(obs, 1)
        persistence = min(obs / max(self.config.recovery_min_track_hits * 2, 8), 1.0)
        quality = float(max(0.0, min(1.0, self.track_quality_ema.get(track_id, 0.0))))
        score = 0.45 * good_ratio + 0.35 * persistence + 0.20 * quality
        return float(max(0.0, min(1.0, score)))

    def _apply_reference_components(
        self,
        components: List[Tuple[float, float]],
        metrics: Dict[str, Any],
        refs: Dict[str, Optional[float]],
        weight_scale: float = 1.0,
    ):
        # ── v58: check sw reliability ──
        sw_reliable = True
        if self.prescan_profiles is not None:
            sw_reliable = self.prescan_profiles.get('sw_reliable', True)

        for key, weight, tolerance in [
            ('sw', 0.11, 0.45),
            ('bh', 0.07, 0.55),
            ('torso', 0.08, 0.55),
            ('size', 0.12, 0.45),
        ]:
            # ── v58: reduce sw weight when prescan flagged it unreliable ──
            effective_weight = weight
            if key == 'sw' and not sw_reliable:
                effective_weight *= 0.30

            value_key = {
                'sw': 'shoulder_width',
                'bh': 'bbox_height',
                'torso': 'torso_height',
                'size': 'size_signature',
            }[key]
            close = metric_closeness(metrics.get(value_key), refs.get(key), tolerance=tolerance)
            if close is not None:
                components.append((close, effective_weight * weight_scale))

    def role_fit_score(
        self,
        person: dict,
        role: str,
        frame: Optional[np.ndarray] = None,
        current_frame: Optional[int] = None,
        use_memory: bool = True,
    ) -> float:
        metrics = self._get_metrics(person)
        components: List[Tuple[float, float]] = []

        human_score = float(person.get('human_score', metrics.get('human_score', 0.0)))
        quality = float(person.get('quality', human_score))
        tid = int(person.get('track_id', -1))
        track_trust = self._track_trust_score(tid)

        components.append((human_score, 0.26))
        components.append((max(0.0, min(1.0, quality)), 0.08))
        components.append((track_trust, 0.06))

        self._apply_reference_components(components, metrics, self._role_refs(role), weight_scale=0.85)
        self._apply_reference_components(components, metrics, self._anchor_refs(role), weight_scale=1.10)

        mem = self.memories.get(role)
        if use_memory and current_frame is not None and mem is not None and mem.is_alive(current_frame):
            pos_score = mem.compare_position(person['center'], current_frame)
            size_score = mem.compare_size(person['kpts'])
            components.append((pos_score, 0.16))
            components.append((size_score, 0.10))
            desc = self._compute_appearance_descriptor(frame, person)
            if desc is not None:
                components.append((mem.compare_appearance(desc), 0.08))
            if mem.bbox_heights:
                bh_close = metric_closeness(metrics.get('bbox_height'), float(np.median(mem.bbox_heights)), tolerance=0.50)
                if bh_close is not None:
                    components.append((bh_close, 0.05))

        total_w = sum(w for _, w in components if w > 0)
        score = (sum(v * w for v, w in components if w > 0) / total_w) if total_w > 0 else human_score

        penalty = 0.0
        penalty += self._role_floor_penalty(metrics, role)
        if current_frame is not None and self.is_role_track_banned(role, tid, current_frame):
            penalty += 0.28
        if tid < 0:
            penalty += 0.05
        if human_score < self.config.human_score_trusted:
            penalty += 0.06

        score = float(max(0.0, min(1.0, score - penalty)))
        return score

    def size_guard_ok(
        self,
        role: str,
        person: dict,
        frame: Optional[np.ndarray] = None,
        current_frame: Optional[int] = None,
    ) -> bool:
        fit = self.role_fit_score(person, role, frame=frame, current_frame=current_frame, use_memory=False)
        if self.prescan_profiles is None:
            return fit >= max(self.config.human_score_trusted, 0.45)
        other = 'CHILD' if role == 'GUARDIAN' else 'GUARDIAN'
        other_fit = self.role_fit_score(person, other, frame=frame, current_frame=current_frame, use_memory=False)
        return fit >= self.config.role_fit_accept_threshold and not (other_fit > fit + self.config.role_fit_margin)

    def accumulate(self, track_id: int, person: dict, frame: Optional[np.ndarray] = None) -> bool:
        if track_id < 0:
            return False
        metrics = self._get_metrics(person)
        if metrics.get('shoulder_width') is None or metrics.get('bbox_height') in (None, 0):
            return False
        if metrics.get('human_score', person.get('human_score', 0.0)) < self.config.human_score_trusted:
            return False

        hist = self.feature_history[track_id]
        if metrics.get('torso_height') is not None:
            hist['torso_height'].append(float(metrics['torso_height']))
        hist['shoulder_width'].append(float(metrics['shoulder_width']))
        hist['body_area'].append(float(metrics['bbox_area']))
        hist['bbox_height'].append(float(metrics['bbox_height']))
        hist['center_y'].append(float(metrics['center_y']))
        hist['size_signature'].append(float(metrics['size_signature']))
        hist['human_score'].append(float(person.get('human_score', metrics.get('human_score', 0.0))))
        hist['quality'].append(float(person.get('quality', metrics.get('human_score', 0.0))))
        hist['det_conf'].append(float(person.get('det_conf', 0.0)))

        if frame is not None and len(hist['bbox_height']) % 5 == 0:
            desc = self._compute_appearance_descriptor(frame, person)
            if desc is not None:
                hist['appearance'].append(desc)

        required = self.config.amostras_para_decisao
        return (
            len(hist['shoulder_width']) >= required and
            len(hist['bbox_height']) >= required and
            len(hist['size_signature']) >= required and
            len(hist['human_score']) >= required
        )

    def _history_summary(self, track_id: int) -> Optional[Dict[str, float]]:
        hist = self.feature_history.get(track_id)
        if not hist or len(hist['shoulder_width']) == 0:
            return None

        def _median(values: deque) -> Optional[float]:
            return float(np.median(values)) if len(values) else None

        return {
            'shoulder_width': _median(hist['shoulder_width']),
            'torso_height': _median(hist['torso_height']),
            'body_area': _median(hist['body_area']),
            'bbox_height': _median(hist['bbox_height']),
            'center_y': _median(hist['center_y']),
            'size_signature': _median(hist['size_signature']),
            'human_score': _median(hist['human_score']),
            'quality': _median(hist['quality']),
            'det_conf': _median(hist['det_conf']),
            'n_samples': float(len(hist['shoulder_width'])),
        }

    def _summary_role_fit(self, summary: Dict[str, float], role: str, track_id: Optional[int] = None) -> float:
        components: List[Tuple[float, float]] = []
        if summary.get('human_score') is not None:
            components.append((float(summary['human_score']), 0.28))
        if summary.get('quality') is not None:
            components.append((float(max(0.0, min(1.0, summary['quality']))), 0.10))
        if track_id is not None:
            components.append((self._track_trust_score(track_id), 0.07))

        summary_metrics = {
            'shoulder_width': summary.get('shoulder_width'),
            'bbox_height': summary.get('bbox_height'),
            'torso_height': summary.get('torso_height'),
            'size_signature': summary.get('size_signature'),
        }
        self._apply_reference_components(components, summary_metrics, self._role_refs(role), weight_scale=0.90)
        self._apply_reference_components(components, summary_metrics, self._anchor_refs(role), weight_scale=1.10)

        total_w = sum(w for _, w in components if w > 0)
        score = (sum(v * w for v, w in components if w > 0) / total_w) if total_w > 0 else float(summary.get('human_score', 0.0))
        score -= self._role_floor_penalty(summary_metrics, role)
        return float(max(0.0, min(1.0, score)))

    def _summary_pair_assignment_score(
        self,
        guardian_summary: Dict[str, float],
        child_summary: Dict[str, float],
        guardian_tid: Optional[int] = None,
        child_tid: Optional[int] = None,
        current_frame: Optional[int] = None,
        apply_bans: bool = True,
    ) -> float:
        components: List[Tuple[float, float]] = []
        g_fit = self._summary_role_fit(guardian_summary, 'GUARDIAN', track_id=guardian_tid)
        c_fit = self._summary_role_fit(child_summary, 'CHILD', track_id=child_tid)
        components.append((g_fit, 0.29))
        components.append((c_fit, 0.29))

        for key, weight, ambiguity, saturation in [
            ('shoulder_width', 0.12, 0.03, 0.28),
            ('torso_height', 0.10, 0.03, 0.28),
            ('size_signature', 0.16, 0.02, 0.34),
            ('bbox_height', 0.05, 0.04, 0.22),
            ('body_area', 0.05, 0.04, 0.28),
        ]:
            ord_score = relative_order_score(
                guardian_summary.get(key), child_summary.get(key), ambiguity=ambiguity, saturation=saturation
            )
            if ord_score is not None:
                components.append((ord_score, weight))

        actual_ratio = safe_ratio(guardian_summary.get('size_signature'), child_summary.get('size_signature'))
        if self.prescan_profiles is not None:
            target_ratio = safe_ratio(self.prescan_profiles.get('guardian_size'), self.prescan_profiles.get('child_size'))
            ratio_close = metric_closeness(actual_ratio, target_ratio, tolerance=0.70)
            if ratio_close is not None:
                components.append((ratio_close, 0.08))
        if self.lock_anchor_profiles is not None:
            anchor_ratio = safe_ratio(
                self.lock_anchor_profiles.get('GUARDIAN', {}).get('size_signature'),
                self.lock_anchor_profiles.get('CHILD', {}).get('size_signature'),
            )
            ratio_close = metric_closeness(actual_ratio, anchor_ratio, tolerance=0.60)
            if ratio_close is not None:
                components.append((ratio_close, 0.08))

        if guardian_tid is not None and child_tid is not None:
            trust_avg = 0.5 * (self._track_trust_score(guardian_tid) + self._track_trust_score(child_tid))
            components.append((trust_avg, 0.04))
            positive_ratio = 0.5 * float(guardian_tid >= 0) + 0.5 * float(child_tid >= 0)
            components.append((positive_ratio, 0.02))

        total_w = sum(w for _, w in components if w > 0)
        avg = (sum(v * w for v, w in components if w > 0) / total_w) if total_w > 0 else 0.0

        penalty = 0.0
        if guardian_tid is not None and child_tid is not None and guardian_tid == child_tid:
            penalty += 1.0
        if apply_bans and current_frame is not None:
            if guardian_tid is not None and self.is_role_track_banned('GUARDIAN', guardian_tid, current_frame):
                penalty += 0.40
            if child_tid is not None and self.is_role_track_banned('CHILD', child_tid, current_frame):
                penalty += 0.40

        summary_guardian_metrics = {
            'shoulder_width': guardian_summary.get('shoulder_width'),
            'bbox_height': guardian_summary.get('bbox_height'),
            'torso_height': guardian_summary.get('torso_height'),
            'size_signature': guardian_summary.get('size_signature'),
        }
        summary_child_metrics = {
            'shoulder_width': child_summary.get('shoulder_width'),
            'bbox_height': child_summary.get('bbox_height'),
            'torso_height': child_summary.get('torso_height'),
            'size_signature': child_summary.get('size_signature'),
        }
        penalty += self._pair_ratio_penalty(summary_guardian_metrics, summary_child_metrics)

        score = float(max(0.0, min(1.0, avg - penalty)))
        return 2.0 * score

    def pair_assignment_score(
        self,
        guardian_person: dict,
        child_person: dict,
        frame: Optional[np.ndarray] = None,
        current_frame: Optional[int] = None,
        use_memory: bool = True,
        apply_bans: bool = True,
    ) -> float:
        components: List[Tuple[float, float]] = []
        g_fit = self.role_fit_score(guardian_person, 'GUARDIAN', frame=frame, current_frame=current_frame, use_memory=use_memory)
        c_fit = self.role_fit_score(child_person, 'CHILD', frame=frame, current_frame=current_frame, use_memory=use_memory)
        components.append((g_fit, 0.29))
        components.append((c_fit, 0.29))

        m_g = self._get_metrics(guardian_person)
        m_c = self._get_metrics(child_person)
        for key, weight, ambiguity, saturation in [
            ('shoulder_width', 0.12, 0.03, 0.28),
            ('torso_height', 0.10, 0.03, 0.28),
            ('size_signature', 0.16, 0.02, 0.34),
            ('bbox_height', 0.05, 0.04, 0.22),
            ('bbox_area', 0.05, 0.04, 0.28),
        ]:
            ord_score = relative_order_score(m_g.get(key), m_c.get(key), ambiguity=ambiguity, saturation=saturation)
            if ord_score is not None:
                components.append((ord_score, weight))

        actual_ratio = safe_ratio(m_g.get('size_signature'), m_c.get('size_signature'))
        if self.prescan_profiles is not None:
            target_ratio = safe_ratio(self.prescan_profiles.get('guardian_size'), self.prescan_profiles.get('child_size'))
            ratio_close = metric_closeness(actual_ratio, target_ratio, tolerance=0.70)
            if ratio_close is not None:
                components.append((ratio_close, 0.08))
        if self.lock_anchor_profiles is not None:
            anchor_ratio = safe_ratio(
                self.lock_anchor_profiles.get('GUARDIAN', {}).get('size_signature'),
                self.lock_anchor_profiles.get('CHILD', {}).get('size_signature'),
            )
            ratio_close = metric_closeness(actual_ratio, anchor_ratio, tolerance=0.60)
            if ratio_close is not None:
                components.append((ratio_close, 0.08))

        g_tid = int(guardian_person.get('track_id', -1))
        c_tid = int(child_person.get('track_id', -1))
        trust_avg = 0.5 * (self._track_trust_score(g_tid) + self._track_trust_score(c_tid))
        components.append((trust_avg, 0.04))
        positive_ratio = 0.5 * float(g_tid >= 0) + 0.5 * float(c_tid >= 0)
        components.append((positive_ratio, 0.02))

        total_w = sum(w for _, w in components if w > 0)
        avg = (sum(v * w for v, w in components if w > 0) / total_w) if total_w > 0 else 0.0

        penalty = 0.0
        if g_tid == c_tid:
            penalty += 1.0
        if apply_bans and current_frame is not None:
            if self.is_role_track_banned('GUARDIAN', g_tid, current_frame):
                penalty += 0.40
            if self.is_role_track_banned('CHILD', c_tid, current_frame):
                penalty += 0.40

        penalty += self._pair_ratio_penalty(m_g, m_c)

        score = float(max(0.0, min(1.0, avg - penalty)))
        return 2.0 * score

    def select_best_lock_pair(self, people: list, current_frame: Optional[int] = None) -> Optional[Tuple[dict, dict]]:
        if len(people) < 2:
            return None
        candidates = sorted(
            people,
            key=lambda p: (p.get('quality', 0.0), p.get('human_score', 0.0), p.get('det_conf', 0.0)),
            reverse=True,
        )[:self.config.candidate_pool_max]
        best = None
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                dist = float(np.linalg.norm(a['center'] - b['center']))
                if dist < self.config.min_pair_separation_px:
                    continue
                score_ab = self.pair_assignment_score(a, b, current_frame=current_frame, use_memory=False)
                score_ba = self.pair_assignment_score(b, a, current_frame=current_frame, use_memory=False)
                best_order_score = max(score_ab, score_ba)
                margin = abs(score_ab - score_ba)
                candidate_score = best_order_score + 0.10 * margin + 0.04 * (a.get('quality', 0.0) + b.get('quality', 0.0))
                if best is None or candidate_score > best[0]:
                    best = (candidate_score, a, b)
        if best is None:
            return None
        return best[1], best[2]

    def decide_and_lock(self, person_a: dict, person_b: dict, frame_id: int):
        id_a = int(person_a['track_id'])
        id_b = int(person_b['track_id'])
        if id_a < 0 or id_b < 0:
            return

        summary_a = self._history_summary(id_a)
        summary_b = self._history_summary(id_b)
        if summary_a is None or summary_b is None:
            return

        def _log_refusal(reason: str):
            self._emit_identity_event(f"\n>>> [Frame {frame_id}] LOCK REFUSED: {reason}")

        trust_a = self._track_trust_score(id_a)
        trust_b = self._track_trust_score(id_b)
        if min(trust_a, trust_b) < 0.25:
            _log_refusal(f"track_trust too low ({trust_a:.2f}, {trust_b:.2f})")
            return

        score_ag_bc = self._summary_pair_assignment_score(summary_a, summary_b, guardian_tid=id_a, child_tid=id_b, current_frame=frame_id, apply_bans=True)
        score_bg_ac = self._summary_pair_assignment_score(summary_b, summary_a, guardian_tid=id_b, child_tid=id_a, current_frame=frame_id, apply_bans=True)
        raw_score_ag_bc = self._summary_pair_assignment_score(summary_a, summary_b, guardian_tid=id_a, child_tid=id_b, current_frame=None, apply_bans=False)
        raw_score_bg_ac = self._summary_pair_assignment_score(summary_b, summary_a, guardian_tid=id_b, child_tid=id_a, current_frame=None, apply_bans=False)
        best_score = max(score_ag_bc, score_bg_ac)
        margin = abs(score_ag_bc - score_bg_ac)
        raw_best_score = max(raw_score_ag_bc, raw_score_bg_ac)
        raw_margin = abs(raw_score_ag_bc - raw_score_bg_ac)

        if score_ag_bc >= score_bg_ac:
            guardian_id, child_id = id_a, id_b
            guardian_summary, child_summary = summary_a, summary_b
            chosen_raw_score = raw_score_ag_bc
        else:
            guardian_id, child_id = id_b, id_a
            guardian_summary, child_summary = summary_b, summary_a
            chosen_raw_score = raw_score_bg_ac

        ambiguous_pair = self._pair_structural_ambiguity(
            {
                'shoulder_width': guardian_summary.get('shoulder_width'),
                'bbox_height': guardian_summary.get('bbox_height'),
                'torso_height': guardian_summary.get('torso_height'),
                'size_signature': guardian_summary.get('size_signature'),
            },
            {
                'shoulder_width': child_summary.get('shoulder_width'),
                'bbox_height': child_summary.get('bbox_height'),
                'torso_height': child_summary.get('torso_height'),
                'size_signature': child_summary.get('size_signature'),
            },
        )

        if best_score < self.adaptive_lock_pair_min_score:
            _log_refusal(f"pair_score={best_score:.2f} < {self.adaptive_lock_pair_min_score:.2f}")
            return
        if margin < self.adaptive_lock_pair_min_margin:
            _log_refusal(f"pair_margin={margin:.2f} < {self.adaptive_lock_pair_min_margin:.2f}")
            return
        if raw_best_score < self.adaptive_lock_pair_min_score:
            _log_refusal(f"raw_pair_score={raw_best_score:.2f} < {self.adaptive_lock_pair_min_score:.2f}")
            return
        if ambiguous_pair and chosen_raw_score < self.config.honeymoon_low_score_threshold:
            if not self.similar_pair_mode:
                _log_refusal(f"pair structurally ambiguous (raw={chosen_raw_score:.2f})")
                return
        if ambiguous_pair and raw_margin < max(0.08, self.adaptive_lock_pair_min_margin * 0.75):
            _log_refusal(f"raw_pair_margin={raw_margin:.2f} too low for ambiguous pair")
            return

        self.identities.clear()
        self.set_role_track('GUARDIAN', guardian_id)
        self.set_role_track('CHILD', child_id)
        self.is_locked = True
        self.lock_frame = frame_id
        self.honeymoon_active = True
        self.honeymoon_frame_count = 0
        self.honeymoon_swap_votes = 0
        self.honeymoon_bad_quality_votes = 0
        self.role_mismatch_streak = {'GUARDIAN': 0, 'CHILD': 0}
        self.inversion_streak = 0
        self.lock_anchor_profiles = {
            'GUARDIAN': dict(guardian_summary),
            'CHILD': dict(child_summary),
        }
        self.last_instability_reason = ''

        self._emit_identity_event(f"\n>>> [Frame {frame_id}] IDENTITY LOCKED v58: {self.identities}")
        self._emit_identity_event(f"    pair_score={best_score:.2f}, margin={margin:.2f}")
        self._emit_identity_event(f"    HONEYMOON: active for next {self.HONEYMOON_FRAMES} frames")

    def resolve_locked_roles(self, people: list, frame: np.ndarray, current_frame: int) -> Tuple[Dict[str, dict], Dict[str, int]]:
        info = {'attempts': 0, 'recovered': 0}
        identified: Dict[str, dict] = {}
        used_idx: set = set()

        for idx, p in enumerate(people):
            tid = int(p['track_id'])
            role = self.identities.get(tid)
            if role and role not in identified:
                if not self.is_role_track_banned(role, tid, current_frame):
                    identified[role] = p
                    used_idx.add(idx)

        missing_roles = [r for r in ['GUARDIAN', 'CHILD'] if r not in identified]
        if not missing_roles:
            return identified, info

        if self.in_uncertainty_cooldown(current_frame):
            info['attempts'] = len(missing_roles)
            return identified, info

        candidates = [people[i] for i in range(len(people)) if i not in used_idx]
        if not candidates:
            info['attempts'] = len(missing_roles)
            return identified, info

        role_scores: Dict[Tuple[str, int], float] = {}
        memory_aux: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
        for role in missing_roles:
            mem = self.memories[role]
            for i, person in enumerate(candidates):
                tid = int(person['track_id'])
                score = self.role_fit_score(person, role, frame=frame, current_frame=current_frame, use_memory=True)
                track_trust = self._track_trust_score(tid)
                mem_pos = mem.compare_position(person['center'], current_frame) if mem.is_alive(current_frame) else 0.0
                mem_size = mem.compare_size(person['kpts']) if mem.is_alive(current_frame) else 0.0
                strong_memory = max(mem_pos, mem_size)
                if tid >= 0 and self.track_observation_count.get(tid, 0) < self.config.recovery_min_track_hits and strong_memory < 0.62:
                    score -= 0.22
                if tid < 0 and strong_memory < 0.72:
                    score -= 0.25
                if track_trust < self.config.recovery_min_track_trust and strong_memory < 0.62:
                    score -= 0.18
                if tid in self.identities and self.identities[tid] != role:
                    score -= 0.30
                if self.is_role_track_banned(role, tid, current_frame):
                    score -= 0.40
                role_scores[(role, i)] = score
                memory_aux[(role, i)] = (track_trust, mem_pos, mem_size)

        info['attempts'] = len(missing_roles)

        if len(missing_roles) == 2 and len(candidates) >= 2:
            combos = []
            for i in range(len(candidates)):
                for j in range(len(candidates)):
                    if i == j:
                        continue
                    pair_score = self.pair_assignment_score(
                        candidates[i], candidates[j], frame=frame, current_frame=current_frame, use_memory=True
                    )
                    combos.append((pair_score, i, j))
            combos.sort(key=lambda x: x[0], reverse=True)
            if combos:
                best_score, gi, ci = combos[0]
                second_score = combos[1][0] if len(combos) > 1 else -999.0
                g_single = role_scores.get(('GUARDIAN', gi), 0.0)
                c_single = role_scores.get(('CHILD', ci), 0.0)
                if (
                    best_score >= self.config.lock_pair_min_score and
                    (best_score - second_score) >= (self.config.role_fit_margin * 0.9) and
                    g_single >= (self.config.role_fit_accept_threshold * 0.90) and
                    c_single >= (self.config.role_fit_accept_threshold * 0.90)
                ):
                    g_person = candidates[gi]
                    c_person = candidates[ci]
                    self.set_role_track('GUARDIAN', int(g_person['track_id']))
                    self.set_role_track('CHILD', int(c_person['track_id']))
                    identified['GUARDIAN'] = g_person
                    identified['CHILD'] = c_person
                    info['recovered'] = 2
                    return identified, info

        for role in missing_roles:
            ranked = []
            for i, person in enumerate(candidates):
                if any(person is already for already in identified.values()):
                    continue
                score = role_scores.get((role, i), 0.0)
                track_trust, mem_pos, mem_size = memory_aux.get((role, i), (0.0, 0.0, 0.0))
                ranked.append((score, i, person, track_trust, mem_pos, mem_size))
            ranked.sort(key=lambda x: x[0], reverse=True)
            if not ranked:
                continue
            best_score, _, best_person, track_trust, mem_pos, mem_size = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else -999.0
            if (
                best_score >= self.config.role_fit_accept_threshold and
                (best_score - second_score) >= (self.config.role_fit_margin * 0.5) and
                (track_trust >= self.config.recovery_min_track_trust or max(mem_pos, mem_size) >= 0.62)
            ):
                self.set_role_track(role, int(best_person['track_id']))
                identified[role] = best_person
                info['recovered'] += 1

        return identified, info

    def evaluate_current_assignment(self, identified: Dict[str, dict], frame: np.ndarray, current_frame: int, ignore_bans: bool = False) -> Tuple[float, float]:
        if 'GUARDIAN' not in identified or 'CHILD' not in identified:
            return 0.0, 0.0
        g_person = identified['GUARDIAN']
        c_person = identified['CHILD']
        eval_frame = None if ignore_bans else current_frame
        current_score = self.pair_assignment_score(g_person, c_person, frame=frame, current_frame=eval_frame, use_memory=False, apply_bans=(not ignore_bans))
        swapped_score = self.pair_assignment_score(c_person, g_person, frame=frame, current_frame=eval_frame, use_memory=False, apply_bans=(not ignore_bans))
        return float(current_score), float(swapped_score)

    def soft_reset(self, clear_histories: bool = False, clear_track_stats: bool = False):
        self.identities.clear()
        if clear_histories:
            self.feature_history.clear()
        if clear_track_stats:
            self.track_last_seen.clear()
            self.track_observation_count.clear()
            self.track_good_count.clear()
            self.track_quality_ema.clear()
            self.role_ban_until.clear()
        self.role_mismatch_streak = {'GUARDIAN': 0, 'CHILD': 0}
        self.is_locked = False
        self.lock_frame = -1
        self.honeymoon_active = False
        self.honeymoon_frame_count = 0
        self.honeymoon_swap_votes = 0
        self.honeymoon_bad_quality_votes = 0
        self.inversion_streak = 0
        self.lock_anchor_profiles = None
        self.pair_ambiguous_grace_until = -1  # v58: reset grace on soft_reset


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 7: VISUALIZAÇÃO# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 7: VISUALIZAÇÃO                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Visualizer:
    COLORS = {
        'GUARDIAN': (200, 100, 0),
        'CHILD':    (0, 180, 80),
        'UNKNOWN':  (128, 128, 128),
        'GHOST_G':  (100, 50, 0),
        'GHOST_K':  (0, 90, 40),
    }

    def __init__(self, config: TitanConfig):
        self.config = config

    def draw_skeleton(self, frame, kpts, color, alpha=1.0):
        if not self.config.draw_skeleton:
            return
        ec = tuple(int(c * alpha) for c in color)
        for i, j in self.config.skeleton_connections:
            if kpts[i][2] > 0.12 and kpts[j][2] > 0.12:
                cv2.line(frame, tuple(kpts[i][:2].astype(int)), tuple(kpts[j][:2].astype(int)), ec, 2, cv2.LINE_AA)
        for kpt in kpts:
            if kpt[2] > 0.12:
                cv2.circle(frame, tuple(kpt[:2].astype(int)), 3, ec, -1, cv2.LINE_AA)

    def draw_gaze(self, frame, origin, gaze_vec, is_looking):
        if not self.config.draw_gaze_vector or gaze_vec is None:
            return
        color = (0, 255, 255) if is_looking else (255, 0, 255)
        thick = 3 if is_looking else 2
        mag = self.config.gaze_vector_magnitude
        start = tuple(origin.astype(int))
        end = (int(origin[0] + gaze_vec[0] * mag), int(origin[1] + gaze_vec[1] * mag))
        cv2.arrowedLine(frame, start, end, color, thick, tipLength=0.3, line_type=cv2.LINE_AA)

    def draw_label(self, frame, role, pos, is_looking, confidence, score):
        color = self.COLORS.get(role, self.COLORS['UNKNOWN'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = int(pos[0]), int(pos[1])

        main_label = role
        sub_label = f"look conf: {confidence:.0%}"
        score_label = f"look score: {score:.3f}"
        status_label = "L@C" if (is_looking and role == 'GUARDIAN') else ("L@G" if (is_looking and role == 'CHILD') else "")

        (w1, h1), b1 = cv2.getTextSize(main_label, font, 0.72, 2)
        (w2, h2), b2 = cv2.getTextSize(sub_label, font, 0.45, 1)
        (w3, h3), b3 = cv2.getTextSize(score_label, font, 0.42, 1)
        width = max(w1, w2, w3) + 12
        height = h1 + h2 + h3 + 26
        top = y - height - 10
        if top < 4:
            top = y + 8
        bottom = top + height

        cv2.rectangle(frame, (x, top), (x + width, bottom), color, -1)
        cv2.putText(frame, main_label, (x + 6, top + h1 + 2), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, sub_label, (x + 6, top + h1 + h2 + 10), font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, score_label, (x + 6, top + h1 + h2 + h3 + 16), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        if status_label:
            cv2.putText(frame, status_label, (x, max(18, top - 6)), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_ghost(self, frame, role, predicted_pos, frames_missing):
        if not self.config.draw_memory_ghosts or predicted_pos is None:
            return
        ghost_color = self.COLORS.get(f'GHOST_{role[0]}', (80, 80, 80))
        center = tuple(predicted_pos.astype(int))
        alpha = max(0.2, 1.0 - frames_missing / 60.0)
        radius = int(25 * alpha)
        cv2.circle(frame, center, radius, ghost_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{role}?", (center[0] - 30, center[1] - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ghost_color, 1, cv2.LINE_AA)

    def draw_connection(self, frame, pos_g, pos_k, g_look, c_look):
        if not self.config.draw_connection_line:
            return
        p1, p2 = tuple(pos_g.astype(int)), tuple(pos_k.astype(int))
        if g_look and c_look:
            color, thick = (0, 255, 0), 3
        elif g_look or c_look:
            color, thick = (0, 200, 255), 2
        else:
            color, thick = (80, 80, 80), 1
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)

    def draw_dashboard(self, frame, stats):
        if not self.config.draw_dashboard:
            return
        h, w = frame.shape[:2]
        pw, ph = 390, 270
        x0, y0 = w - pw - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "TITAN v58 GAIA", (x0 + 10, y0 + 22),
                    font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        y = y0 + 48
        role_state = 'UNCERTAIN' if stats.get('assignment_uncertain') else 'stable'
        lines = [
            f"Frame: {stats.get('frame_id','?')} / {stats.get('total','?')}",
            f"Detected: {stats.get('n_detected','?')} | Valid: {stats.get('n_valid','?')}",
            f"ID Lock: {'YES' if stats.get('id_locked') else 'NO'}" + (' [HONEYMOON]' if stats.get('honeymoon') else ''),
            f"Role state: {role_state}",
            f"G vis: {'YES' if stats.get('g_visible') else 'no'} | look: {stats.get('g_score',0):.3f} {'L@C' if stats.get('g_looking') else ''}",
            f"C vis: {'YES' if stats.get('k_visible') else 'no'} | look: {stats.get('c_score',0):.3f} {'L@G' if stats.get('c_looking') else ''}",
            f"G mem: {stats.get('g_mem_frames','?')}f | C mem: {stats.get('k_mem_frames','?')}f",
            f"Data: {'RECORDING' if stats.get('recording') else 'DISCARDED'}",
            f"Reason: {stats.get('uncertainty_reason', '-')}",
            f"GPU: {stats.get('gpu_name','CPU')} | VRAM: {stats.get('vram_used','?')}MB",
            f"Iris gaze: {'ACTIVE' if MEDIAPIPE_AVAILABLE else 'DISABLED'}",
        ]
        for line in lines:
            color = (200, 200, 200)
            if 'DISCARDED' in line or 'UNCERTAIN' in line:
                color = (0, 170, 255) if 'UNCERTAIN' in line else (0, 0, 200)
            elif 'RECORDING' in line:
                color = (0, 220, 0)
            elif 'HONEYMOON' in line:
                color = (0, 200, 255)
            cv2.putText(frame, line, (x0 + 10, y), font, 0.42, color, 1, cv2.LINE_AA)
            y += 22


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 8: UTILIDADES                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def calc_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    v = p2 - p1
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 9: PIPELINE PRINCIPAL                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def run_prescan(config: TitanConfig, model, device, use_half):
    print(f"\n{'='*70}")
    print("  PRE-SCAN: Establishing body size profiles...")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        print("    [PRESCAN] WARNING: Could not open video.")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_start = int(config.segundos_para_pular * fps_vid)
    stride = max(1, int(config.prescan_stride_frames))

    all_measurements = []
    pair_big = []
    pair_small = []
    rejected = 0
    sampled = 0
    t0 = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < frame_start:
            frame_idx += 1
            continue
        if (frame_idx - frame_start) % stride != 0:
            frame_idx += 1
            continue

        results = model.predict(
            frame, verbose=False, classes=[0],
            conf=0.25, iou=0.60, half=use_half,
            imgsz=config.imgsz, device=device,
        )

        frame_candidates = []
        if results and results[0].keypoints is not None and results[0].boxes is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            confs = results[0].boxes.conf.cpu().numpy() if getattr(results[0].boxes, 'conf', None) is not None else np.ones(len(keypoints), dtype=np.float32)

            for i, kpts in enumerate(keypoints):
                box = boxes[i] if i < len(boxes) else None
                det_conf = float(confs[i]) if i < len(confs) else 0.0
                ok, human_score, metrics = score_human_pose(kpts, box, det_conf, mode='extra', config=config)
                if not ok:
                    rejected += 1
                    continue
                if metrics.get('shoulder_width') is None or metrics.get('bbox_height') in (None, 0):
                    rejected += 1
                    continue
                measurement = {
                    'shoulder_width': float(metrics['shoulder_width']),
                    'bbox_height': float(metrics['bbox_height']),
                    'torso_height': float(metrics['torso_height']) if metrics.get('torso_height') is not None else None,
                    'size_signature': float(metrics['size_signature']),
                    'human_score': float(human_score),
                }
                all_measurements.append(measurement)
                frame_candidates.append(measurement)

        if len(frame_candidates) >= 2:
            frame_candidates.sort(key=lambda m: (m['human_score'], m['size_signature']), reverse=True)
            top2 = frame_candidates[:2]
            top2.sort(key=lambda m: m['size_signature'], reverse=True)
            pair_big.append(top2[0])
            pair_small.append(top2[1])

        sampled += 1
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"    Sampled {sampled} frames in {elapsed:.1f}s ({len(all_measurements)} valid, {rejected} rejected, {len(pair_big)} paired)")

    profiles = None
    if len(pair_big) >= 8 and len(pair_small) >= 8:
        profiles = {
            'guardian_sw': float(np.median([m['shoulder_width'] for m in pair_big])),
            'child_sw': float(np.median([m['shoulder_width'] for m in pair_small])),
            'guardian_bh': float(np.median([m['bbox_height'] for m in pair_big])),
            'child_bh': float(np.median([m['bbox_height'] for m in pair_small])),
            'guardian_torso': float(np.median([m['torso_height'] for m in pair_big if m['torso_height'] is not None])) if any(m['torso_height'] is not None for m in pair_big) else None,
            'child_torso': float(np.median([m['torso_height'] for m in pair_small if m['torso_height'] is not None])) if any(m['torso_height'] is not None for m in pair_small) else None,
            'guardian_size': float(np.median([m['size_signature'] for m in pair_big])),
            'child_size': float(np.median([m['size_signature'] for m in pair_small])),
        }
        profiles['sw_midpoint'] = (profiles['guardian_sw'] + profiles['child_sw']) / 2.0
        profiles['bh_midpoint'] = (profiles['guardian_bh'] + profiles['child_bh']) / 2.0
        profiles['size_ratio'] = profiles['guardian_size'] / max(profiles['child_size'], 1.0)
        # ── v58: flag sw reliability ──
        profiles['sw_reliable'] = bool(profiles['guardian_sw'] > profiles['child_sw'])
        print(f"    → PAIR-BASED prescan ({len(pair_big)} paired frames)")
        if not profiles['sw_reliable']:
            print(f"    ⚠ WARNING: child_sw > guardian_sw ({profiles['child_sw']:.0f} > {profiles['guardian_sw']:.0f}) — sw weight reduced")

    if profiles is None and len(all_measurements) >= 20:
        values = sorted(m['size_signature'] for m in all_measurements)
        n = len(values)
        lo = int(n * 0.2)
        hi = int(n * 0.8)
        best_gap = -1.0
        best_cut = n // 2
        for i in range(lo, max(lo + 1, hi - 1)):
            gap = values[i + 1] - values[i]
            if gap > best_gap:
                best_gap = gap
                best_cut = i + 1
        split = values[max(0, min(best_cut - 1, n - 1))]
        cluster_small = [m for m in all_measurements if m['size_signature'] <= split]
        cluster_big = [m for m in all_measurements if m['size_signature'] > split]
        if len(cluster_small) >= 5 and len(cluster_big) >= 5:
            profiles = {
                'guardian_sw': float(np.median([m['shoulder_width'] for m in cluster_big])),
                'child_sw': float(np.median([m['shoulder_width'] for m in cluster_small])),
                'guardian_bh': float(np.median([m['bbox_height'] for m in cluster_big])),
                'child_bh': float(np.median([m['bbox_height'] for m in cluster_small])),
                'guardian_torso': float(np.median([m['torso_height'] for m in cluster_big if m['torso_height'] is not None])) if any(m['torso_height'] is not None for m in cluster_big) else None,
                'child_torso': float(np.median([m['torso_height'] for m in cluster_small if m['torso_height'] is not None])) if any(m['torso_height'] is not None for m in cluster_small) else None,
                'guardian_size': float(np.median([m['size_signature'] for m in cluster_big])),
                'child_size': float(np.median([m['size_signature'] for m in cluster_small])),
            }
            profiles['sw_midpoint'] = (profiles['guardian_sw'] + profiles['child_sw']) / 2.0
            profiles['bh_midpoint'] = (profiles['guardian_bh'] + profiles['child_bh']) / 2.0
            profiles['size_ratio'] = profiles['guardian_size'] / max(profiles['child_size'], 1.0)
            # ── v58: flag sw reliability ──
            profiles['sw_reliable'] = bool(profiles['guardian_sw'] > profiles['child_sw'])
            print(f"    → GAP CLUSTERING fallback ({len(cluster_big)}/{len(cluster_small)} clusters)")
            if not profiles['sw_reliable']:
                print(f"    ⚠ WARNING: child_sw > guardian_sw ({profiles['child_sw']:.0f} > {profiles['guardian_sw']:.0f}) — sw weight reduced")

    if profiles is None:
        print("    [PRESCAN] WARNING: Could not establish profiles.")
        return None

    print("\n    PRESCAN PROFILES (v58):")
    print(f"    GUARDIAN: sw={profiles['guardian_sw']:.0f}px, bh={profiles['guardian_bh']:.0f}px, size={profiles['guardian_size']:.0f}")
    print(f"    CHILD:    sw={profiles['child_sw']:.0f}px, bh={profiles['child_bh']:.0f}px, size={profiles['child_size']:.0f}")
    print(f"    MIDPOINT: sw={profiles['sw_midpoint']:.0f}px | bh={profiles['bh_midpoint']:.0f}px")
    print(f"    RATIO:    {profiles['size_ratio']:.2f}")
    print(f"{'='*70}\n")
    return profiles


def run_gaia(config: TitanConfig):

    print("=" * 70)
    print("  TITAN v58 — GAIA")
    print("  Gaze Analysis for Interaction Assessment")
    print("=" * 70)

    gpu_info = diagnose_gpu()
    print_gpu_banner(gpu_info)

    n_cores = maximize_cpu()
    print(f">>> CPU: {n_cores} cores — configurado para throughput estável")

    force_cpu = (config.device == 'cpu')
    if force_cpu:
        gpu_info['cuda_available'] = False

    if gpu_info['cuda_available']:
        if config.device not in ('auto', 'cpu') and config.device not in (0, '0', 'cuda:0'):
            try:
                device = int(str(config.device).replace('cuda:', ''))
            except ValueError:
                device = 0
        else:
            device = 0

        if config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(">>> VRAM preallocation agressiva DESATIVADA para evitar instabilidade e fragmentação.")
    else:
        device = 'cpu'

    print(f"\n>>> Carregando modelo {config.model_name}...")
    model = YOLO(config.model_name)
    use_half = bool(config.use_fp16 and gpu_info['cuda_available'] and gpu_info['fp16_supported'])
    if use_half:
        print(">>> FP16 ENABLED")
    print(f">>> Resolução YOLO: {config.imgsz}px | Passes: {len(config.detection_passes)}")

    print(">>> Warmup...")
    if gpu_info['cuda_available']:
        torch.cuda.reset_peak_memory_stats()
    dummy = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, verbose=False, half=use_half, imgsz=config.imgsz, device=device)
    if gpu_info['cuda_available']:
        torch.cuda.synchronize()
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        print(f">>> VRAM peak warmup: {vram_peak:.0f} MB")
    print(">>> Warmup complete.")

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    out_video = os.path.join(script_dir, config.output_video)
    out_json = os.path.join(script_dir, config.output_json)
    out_report = os.path.join(script_dir, config.output_report)
    out_log = os.path.join(script_dir, config.output_video.replace('.mp4', '_debug.log'))
    for p in [out_video, out_json, out_report]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    debug_log = open(out_log, 'w', encoding='utf-8')
    def dlog(msg: str, echo: Optional[bool] = None):
        if echo is None:
            echo = config.console_debug_events
        if echo:
            try:
                tqdm.write(msg)
            except Exception:
                print(msg, flush=True)
        debug_log.write(msg + '\n')
        if config.debug_log_flush_immediate:
            debug_log.flush()

    prescan_profiles = run_prescan(config, model, device, use_half)
    if prescan_profiles:
        dlog(f"\n{'='*70}\nPRESCAN PROFILES v58:")
        for k, v in prescan_profiles.items():
            dlog(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
        dlog(f"{'='*70}\n")
    else:
        dlog("PRESCAN: No profiles established!")

    model.predictor = None
    model.predict(dummy, verbose=False, half=use_half, imgsz=config.imgsz, device=device)

    if not os.path.exists(config.input_video):
        sys.exit(f"FATAL ERROR: Video not found: '{config.input_video}'")

    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        sys.exit("FATAL ERROR: Could not open video.")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> Vídeo: {fw}x{fh} @ {fps}fps, {total_frames} frames")

    frame_start = int(config.segundos_para_pular * fps)
    if not (0 < frame_start < total_frames):
        frame_start = 0
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        print(f">>> Pulando {config.segundos_para_pular}s ({frame_start} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        writer = ThreadedVideoWriter(out_video, fourcc, fps, (fw, fh))
        print(">>> I/O: ThreadedVideoWriter")
    except Exception:
        writer = cv2.VideoWriter(out_video, fourcc, fps, (fw, fh))
        print(">>> I/O: cv2.VideoWriter (fallback)")

    detector = MultiPassDetector(model, config, device, use_half)
    id_mgr = IdentityManager(config)
    id_mgr.event_logger = dlog
    id_mgr.prescan_profiles = prescan_profiles
    if prescan_profiles is not None:
        ratio = prescan_profiles.get('size_ratio', 1.5)
        if ratio < 1.35:
            id_mgr.INTRUDER_CONFIRM_FRAMES = 24
            id_mgr.INVERSION_SWAP_FRAMES = 26
            # ── v58: adaptive lock thresholds for similar pairs ──
            id_mgr.adaptive_lock_pair_min_score = 0.68
            id_mgr.adaptive_lock_pair_min_margin = 0.06
            id_mgr.similar_pair_mode = True
            dlog(f"[ADAPTIVE v58] Similar sizes (ratio={ratio:.2f}) → relaxed lock thresholds (score≥0.68, margin≥0.06)")
        else:
            id_mgr.adaptive_lock_pair_min_score = config.lock_pair_min_score
            id_mgr.adaptive_lock_pair_min_margin = config.lock_pair_min_margin
            id_mgr.similar_pair_mode = False
            dlog(f"[ADAPTIVE v58] Distinct sizes (ratio={ratio:.2f}) → normal lock thresholds")

    smoother = TemporalSmoother(config)
    gaze_est = GazeEstimator()
    print("  ✓ GazeEstimator v58 (crop mais contido + confiança dinâmica)")
    vis = Visualizer(config)

    json_file = open(out_json, 'w', encoding='utf-8')
    json_file.write('[\n')
    json_frames_written = 0

    stats_counter = {
        'total_processed': 0,
        'frames_with_data': 0,
        'frames_discarded': 0,
        'reid_attempts': 0,
        'reid_successes': 0,
        'multipass_used': 0,
        'g_looking_frames': 0,
        'c_looking_frames': 0,
        'mutual_attention_frames': 0,
        'honeymoon_resets': 0,
        'swap_events': 0,
    }

    last_vram_mb = 0
    current_fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    fatal_error = None
    recent_valid_counts = deque(maxlen=90)
    recent_crowd_counts = deque(maxlen=60)
    gaze_error_last_report: Dict[str, int] = {}

    gpu_name_short = gpu_info['device_name'].split(' ')[-1] if gpu_info['cuda_available'] else 'CPU'
    dev_label = f"GPU 0 ({gpu_info['device_name']})" if device == 0 else "CPU"
    print(f"\n>>> GAIA v58 ACTIVE. Device: {dev_label} | FP16: {use_half}")
    print("    v58: pair-based role scoring + role bans + uncertainty cooldown + anti-ping-pong lock + cleaner labels\n")

    try:
        progress = tqdm(
            total=(total_frames - frame_start),
            unit='frame',
            desc="GAIA",
            file=sys.stdout,
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.5,
            maxinterval=2.0,
        )
    except OSError:
        class FakeProgress:
            def __init__(self, total): self.n = 0; self.total = total
            def update(self, n=1):
                self.n += n
                if self.n % 100 == 0:
                    print(f"  Frame {self.n}/{self.total}", flush=True)
            def set_postfix(self, d): pass
            def close(self): pass
        progress = FakeProgress(total_frames - frame_start)

    t_start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        try:
            current_frame = progress.n + frame_start
            stats_counter['total_processed'] += 1

            fps_counter += 1
            if fps_counter >= 10:
                now = time.time()
                current_fps = fps_counter / max(now - fps_timer, 0.001)
                fps_timer = now
                fps_counter = 0
                progress.set_postfix({'fps': f'{current_fps:.1f}', 'dev': 'GPU' if device == 0 else 'CPU'})

            if id_mgr.is_locked:
                detector.known_ids = {tid for tid in id_mgr.identities.keys() if tid >= 0}
            people, raw_results = detector.detect(frame, prefer_recovery=id_mgr.is_locked)

            n_detected = len(raw_results[0].boxes) if raw_results and raw_results[0].boxes is not None else 0
            n_valid = len(people)
            recent_valid_counts.append(n_valid)
            recent_crowd_counts.append(sum(1 for p in people if p.get('human_score', 0.0) >= (config.human_score_strict + 0.08)))
            if detector.last_used_extra_passes:
                stats_counter['multipass_used'] += 1

            # ── REGRA DE OURO v59: gate HARD de exatamente 2 pessoas ──
            exactly_two_valid = (n_valid == 2)

            id_mgr.register_visible_tracks(people, current_frame)
            active_tids = {p['track_id'] for p in people if p['track_id'] >= 0}
            id_mgr.cleanup_stale_ids(active_tids, current_frame)

            # Só acumula features quando exatamente 2 pessoas válidas
            # (previne contaminação por 3ª pessoa ou dados espúrios de 1 pessoa)
            if exactly_two_valid:
                for person in people:
                    tid = int(person['track_id'])
                    if tid >= 0:
                        id_mgr.accumulate(tid, person, frame)

            if exactly_two_valid and id_mgr.can_attempt_lock(current_frame):
                crowding = sum(1 for c in recent_crowd_counts if c >= 4) > max(3, len(recent_crowd_counts) * 0.25)
                pair = None if crowding else id_mgr.select_best_lock_pair(people, current_frame=current_frame)
                if pair is not None:
                    person_a, person_b = pair
                    if person_a['track_id'] >= 0 and person_b['track_id'] >= 0:
                        sum_a = id_mgr._history_summary(person_a['track_id'])
                        sum_b = id_mgr._history_summary(person_b['track_id'])
                        # ── v58: accelerated re-lock when memories alive ──
                        both_memories_alive = all(
                            id_mgr.memories[r].is_alive(current_frame) and id_mgr.memories[r].frames_visible > 30
                            for r in ['GUARDIAN', 'CHILD']
                        )
                        effective_samples = config.amostras_para_decisao // 2 if both_memories_alive else config.amostras_para_decisao
                        ready_a = bool(sum_a and sum_a.get('n_samples', 0) >= effective_samples)
                        ready_b = bool(sum_b and sum_b.get('n_samples', 0) >= effective_samples)
                        if ready_a and ready_b:
                            dlog(f"\n[Frame {current_frame}] LOCK ATTEMPT v58:")
                            dlog(
                                f"  A(tid={person_a['track_id']}): sw={sum_a.get('shoulder_width', 0):.1f}px, "
                                f"bh={sum_a.get('bbox_height', 0):.1f}px, size={sum_a.get('size_signature', 0):.1f}, "
                                f"h={sum_a.get('human_score', 0):.2f}, trust={id_mgr._track_trust_score(int(person_a['track_id'])):.2f}"
                            )
                            dlog(
                                f"  B(tid={person_b['track_id']}): sw={sum_b.get('shoulder_width', 0):.1f}px, "
                                f"bh={sum_b.get('bbox_height', 0):.1f}px, size={sum_b.get('size_signature', 0):.1f}, "
                                f"h={sum_b.get('human_score', 0):.2f}, trust={id_mgr._track_trust_score(int(person_b['track_id'])):.2f}"
                            )
                            was_locked = id_mgr.is_locked
                            id_mgr.decide_and_lock(person_a, person_b, current_frame)
                            if id_mgr.is_locked and not was_locked:
                                dlog(f"  → LOCKED: {id_mgr.identities}")
                            elif not id_mgr.is_locked:
                                dlog("  → REJECTED (aguardando separação melhor e mais histórico)")

            identified: Dict[str, dict] = {}
            current_pair_score = 0.0
            swapped_pair_score = 0.0
            if exactly_two_valid and id_mgr.is_locked:
                identified, resolve_info = id_mgr.resolve_locked_roles(people, frame, current_frame)
                stats_counter['reid_attempts'] += resolve_info.get('attempts', 0)
                stats_counter['reid_successes'] += resolve_info.get('recovered', 0)

                provisional_both = ('GUARDIAN' in identified and 'CHILD' in identified)
                if provisional_both:
                    current_pair_score, swapped_pair_score = id_mgr.evaluate_current_assignment(identified, frame, current_frame, ignore_bans=True)

                stripped_roles = []
                for role, person in list(identified.items()):
                    fit = id_mgr.role_fit_score(person, role, frame=frame, current_frame=current_frame, use_memory=False)
                    other_role = 'CHILD' if role == 'GUARDIAN' else 'GUARDIAN'
                    other_fit = id_mgr.role_fit_score(person, other_role, frame=frame, current_frame=current_frame, use_memory=False)
                    hard_bad = fit < (config.role_fit_accept_threshold * 0.68)
                    weak_fit = fit < (config.role_fit_accept_threshold * 0.82)
                    cross_role_risk = other_fit > fit + config.role_fit_margin
                    weak_pair_context = (not provisional_both) or ((current_pair_score - swapped_pair_score) < (config.role_fit_margin * 0.50))
                    if hard_bad or (cross_role_risk and weak_fit and weak_pair_context):
                        id_mgr.role_mismatch_streak[role] += 1
                    else:
                        id_mgr.role_mismatch_streak[role] = max(0, id_mgr.role_mismatch_streak[role] - 1)
                    if id_mgr.role_mismatch_streak[role] >= id_mgr.INTRUDER_CONFIRM_FRAMES:
                        stripped_roles.append((role, person, fit, other_fit))

                for role, person, fit, other_fit in stripped_roles:
                    identified.pop(role, None)
                    id_mgr.reject_role_candidate(
                        role,
                        person,
                        current_frame,
                        reason=f"fit inconsistente ({fit:.2f} vs {other_fit:.2f})",
                    )
                    if person is not None:
                        dlog(
                            f"[Frame {current_frame}] INTRUDER GUARD: banindo {role} tid={person['track_id']} "
                            f"(fit={fit:.2f}, other={other_fit:.2f})"
                        )

            both_present = ('GUARDIAN' in identified and 'CHILD' in identified)
            assignment_uncertain = id_mgr.in_uncertainty_cooldown(current_frame)
            uncertainty_reason = id_mgr.last_instability_reason if assignment_uncertain else ''

            if id_mgr.honeymoon_active and both_present:
                current_pair_score, swapped_pair_score = id_mgr.evaluate_current_assignment(identified, frame, current_frame, ignore_bans=True)
                pair_ambiguous = id_mgr.current_pair_structurally_ambiguous(identified)
                if swapped_pair_score > current_pair_score + config.inversion_suspect_margin:
                    id_mgr.honeymoon_swap_votes += 1
                if pair_ambiguous or current_pair_score < config.honeymoon_low_score_threshold:
                    id_mgr.honeymoon_bad_quality_votes += 1
                id_mgr.honeymoon_frame_count += 1
                if id_mgr.honeymoon_frame_count >= id_mgr.HONEYMOON_FRAMES:
                    id_mgr.honeymoon_active = False
                    swap_rate = id_mgr.honeymoon_swap_votes / max(id_mgr.honeymoon_frame_count, 1)
                    bad_rate = id_mgr.honeymoon_bad_quality_votes / max(id_mgr.honeymoon_frame_count, 1)
                    dlog(
                        f"[HONEYMOON END frame {current_frame}] swap_rate={swap_rate:.0%} "
                        f"bad_rate={bad_rate:.0%} ({id_mgr.honeymoon_swap_votes}/{id_mgr.honeymoon_frame_count})"
                    )
                    if swap_rate > id_mgr.HONEYMOON_SWAP_THRESHOLD or (
                        not id_mgr.similar_pair_mode and bad_rate > config.honeymoon_bad_vote_threshold
                    ):
                        dlog("  ★ BAD LOCK DETECTED — banindo papeis atuais e forçando relock")
                        id_mgr.reset_after_instability(
                            current_frame,
                            'bad honeymoon lock',
                            identified=identified,
                            ban_current_roles=True,
                            extra_ban_frames=max(12, config.role_uncertainty_cooldown_frames),
                        )
                        smoother.reset()
                        identified.clear()
                        both_present = False
                        assignment_uncertain = True
                        uncertainty_reason = 'bad honeymoon lock'
                        stats_counter['honeymoon_resets'] += 1
                    else:
                        dlog(f"  ✓ LOCK VALIDATED (swap_rate={swap_rate:.0%}, bad_rate={bad_rate:.0%})")

            if id_mgr.is_locked and both_present:
                current_pair_score, swapped_pair_score = id_mgr.evaluate_current_assignment(identified, frame, current_frame, ignore_bans=True)
                pair_ambiguous = id_mgr.current_pair_structurally_ambiguous(identified)
                inversion_delta = swapped_pair_score - current_pair_score
                if inversion_delta > config.inversion_reset_margin:
                    id_mgr.inversion_streak += 1
                    assignment_uncertain = True
                    uncertainty_reason = 'role inversion suspected'
                    if id_mgr.inversion_streak >= max(3, id_mgr.INVERSION_SWAP_FRAMES // 3):
                        id_mgr.enter_uncertainty_cooldown(
                            current_frame,
                            uncertainty_reason,
                            frames=max(8, config.role_uncertainty_cooldown_frames),
                        )
                    if id_mgr.inversion_streak >= id_mgr.INVERSION_SWAP_FRAMES:
                        dlog(
                            f"[Frame {current_frame}] ROLE INSTABILITY: current={current_pair_score:.2f} "
                            f"swapped={swapped_pair_score:.2f} -> reset + relock"
                        )
                        id_mgr.reset_after_instability(
                            current_frame,
                            'persistent inversion',
                            identified=identified,
                            ban_current_roles=True,
                            extra_ban_frames=max(12, config.role_uncertainty_cooldown_frames),
                        )
                        smoother.reset()
                        identified.clear()
                        both_present = False
                        assignment_uncertain = True
                        uncertainty_reason = 'persistent inversion'
                        stats_counter['swap_events'] += 1
                elif inversion_delta > config.inversion_suspect_margin:
                    assignment_uncertain = True
                    uncertainty_reason = 'role inversion suspected'
                    id_mgr.inversion_streak = max(1, id_mgr.inversion_streak)
                    id_mgr.enter_uncertainty_cooldown(
                        current_frame,
                        uncertainty_reason,
                        frames=max(6, config.role_uncertainty_cooldown_frames // 2),
                    )
                elif pair_ambiguous and current_pair_score < config.honeymoon_low_score_threshold:
                    # ── v58: anti-cascade — in similar_pair_mode, skip repeated cooldowns
                    #    if same track_ids are locked and no actual inversion is detected ──
                    if id_mgr.similar_pair_mode and current_frame < id_mgr.pair_ambiguous_grace_until:
                        # Grace period active: accept data, don't re-enter cooldown
                        id_mgr.inversion_streak = max(0, id_mgr.inversion_streak - 1)
                    elif id_mgr.similar_pair_mode and inversion_delta <= config.inversion_suspect_margin:
                        # Similar pair, no inversion — set grace period instead of cascading cooldown
                        assignment_uncertain = True
                        uncertainty_reason = 'pair ambiguous (grace set)'
                        id_mgr.inversion_streak = max(0, id_mgr.inversion_streak - 1)
                        id_mgr.pair_ambiguous_grace_until = current_frame + 30
                        id_mgr.enter_uncertainty_cooldown(
                            current_frame,
                            uncertainty_reason,
                            frames=max(6, config.role_uncertainty_cooldown_frames // 2),
                        )
                    else:
                        assignment_uncertain = True
                        uncertainty_reason = 'pair ambiguous'
                        id_mgr.inversion_streak = max(0, id_mgr.inversion_streak - 1)
                        id_mgr.enter_uncertainty_cooldown(
                            current_frame,
                            uncertainty_reason,
                            frames=max(6, config.role_uncertainty_cooldown_frames // 2),
                        )
                else:
                    id_mgr.inversion_streak = max(0, id_mgr.inversion_streak - 1)

            # ── v59: GATE HARD — só grava dados quando EXATAMENTE 2 pessoas válidas ──
            recording = (exactly_two_valid and both_present and not assignment_uncertain) or (not config.strict_two_person)

            # Só atualiza memórias persistentes quando frame é válido para gravação
            # (exatamente 2 pessoas, ambas identificadas, sem incerteza)
            if exactly_two_valid and not assignment_uncertain:
                for role, person in identified.items():
                    fit = id_mgr.role_fit_score(person, role, frame=frame, current_frame=current_frame, use_memory=False)
                    if fit >= (config.role_fit_accept_threshold * 0.80):
                        id_mgr.memories[role].update(person['kpts'], person['track_id'], current_frame, frame)

            g_looking, c_looking = False, False
            g_score, c_score = 0.0, 0.0
            g_gaze, c_gaze = None, None
            g_conf, c_conf = 0.0, 0.0

            def analyze_gaze(observer_role, target_role):
                if observer_role not in identified:
                    smoother.reset(observer_role)
                    return False, 0.0, None, 0.0
                obs = identified[observer_role]
                target_pos = identified[target_role]['center'] if target_role in identified else id_mgr.memories[target_role].predict_position(current_frame)
                if target_pos is None:
                    looking, score = smoother.update(observer_role, 0.0)
                    return looking, score, None, 0.0
                try:
                    raw_score, confidence = gaze_est.estimate_looking(
                        obs['kpts'], target_pos, frame=frame, role=observer_role,
                        mediapipe_min_yolo_conf=config.mediapipe_min_yolo_face_conf,
                        camera_y_weight=config.camera_y_weight,
                    )
                except Exception as e:
                    last_report = gaze_error_last_report.get(observer_role, -10**9)
                    if (current_frame - last_report) >= config.console_gaze_error_cooldown_frames:
                        dlog(f"[GAZE ERROR frame {current_frame}] {observer_role}: {e}", echo=config.console_gaze_errors)
                        gaze_error_last_report[observer_role] = current_frame
                    raw_score, confidence = 0.0, 0.0
                weighted = raw_score * confidence
                looking, score = smoother.update(observer_role, weighted)
                mp_info = gaze_est._last_mp_results.get(observer_role, {})
                mp_gaze = mp_info.get('gaze_vec')
                gaze_vec = mp_gaze if mp_gaze is not None else gaze_est.estimate(obs['kpts'])
                return looking, score, gaze_vec, confidence

            g_looking, g_score, g_gaze, g_conf = analyze_gaze('GUARDIAN', 'CHILD')
            c_looking, c_score, c_gaze, c_conf = analyze_gaze('CHILD', 'GUARDIAN')

            if g_looking and recording:
                stats_counter['g_looking_frames'] += 1
            if c_looking and recording:
                stats_counter['c_looking_frames'] += 1
            if g_looking and c_looking and recording:
                stats_counter['mutual_attention_frames'] += 1

            # ── v58: face-only privacy blur (preserva linguagem corporal) ──
            if config.privacy_blur:
                k = config.privacy_blur_kernel
                for person in people:
                    kpts = person['kpts']
                    # Coleta keypoints da cabeça visíveis (COCO 0-4: nose, eyes, ears)
                    head_pts = []
                    for idx in [0, 1, 2, 3, 4]:
                        if kpts[idx][2] > 0.04:
                            head_pts.append(kpts[idx][:2])

                    if not head_pts:
                        continue

                    head_pts = np.array(head_pts, dtype=np.float32)
                    cx, cy = np.mean(head_pts, axis=0)

                    # Escala baseada em shoulder_width ou spread dos pontos da cabeça
                    shoulder_width = None
                    if kpts[5][2] > 0.06 and kpts[6][2] > 0.06:
                        shoulder_width = float(np.linalg.norm(kpts[6][:2] - kpts[5][:2]))

                    spread = float(np.max(np.ptp(head_pts, axis=0))) if len(head_pts) >= 2 else 0.0
                    if shoulder_width is not None:
                        radius = max(spread * 1.2, shoulder_width * 0.75, 40.0)
                    else:
                        radius = max(spread * 1.6, 50.0)

                    # Elipse vertical (rosto é mais alto que largo)
                    rx = int(radius * 0.75)
                    ry = int(radius * 1.0)
                    h_frame, w_frame = frame.shape[:2]
                    x1 = max(0, int(cx - rx))
                    y1 = max(0, int(cy - ry))
                    x2 = min(w_frame, int(cx + rx))
                    y2 = min(h_frame, int(cy + ry))

                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    # Aplica blur na região retangular
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)

                    # Máscara elíptica pra blend suave
                    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    center_local = (int(cx - x1), int(cy - y1))
                    cv2.ellipse(mask, center_local, (rx, ry), 0, 0, 360, 255, -1)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    frame[y1:y2, x1:x2] = np.where(mask_3ch > 0, blurred_roi, roi)

            for role, person in identified.items():
                color = Visualizer.COLORS.get(role, (128, 128, 128))
                vis.draw_skeleton(frame, person['kpts'], color)

            for role in ['GUARDIAN', 'CHILD']:
                if role in identified:
                    color = Visualizer.COLORS.get(role, (128, 128, 128))
                    gaze_est.draw_mediapipe(frame, role, color)

            for role in ['GUARDIAN', 'CHILD']:
                if role not in identified:
                    mem = id_mgr.memories[role]
                    if mem.is_alive(current_frame):
                        vis.draw_ghost(frame, role, mem.predict_position(current_frame), mem.frames_missing(current_frame))

            if both_present:
                vis.draw_connection(frame, identified['GUARDIAN']['kpts'][0][:2], identified['CHILD']['kpts'][0][:2], g_looking, c_looking)

            for role, gaze, looking, conf, score in [
                ('GUARDIAN', g_gaze, g_looking, g_conf, g_score),
                ('CHILD', c_gaze, c_looking, c_conf, c_score),
            ]:
                if role in identified:
                    pos = identified[role]['kpts'][0][:2]
                    vis.draw_gaze(frame, pos, gaze, looking)
                    vis.draw_label(frame, role, pos, looking, conf, score)

            if gpu_info['cuda_available'] and current_frame % 30 == 0:
                last_vram_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2))

            vis.draw_dashboard(frame, {
                'frame_id': current_frame,
                'total': total_frames,
                'n_detected': n_detected,
                'n_valid': n_valid,
                'id_locked': id_mgr.is_locked,
                'honeymoon': id_mgr.honeymoon_active,
                'g_visible': 'GUARDIAN' in identified,
                'k_visible': 'CHILD' in identified,
                'g_looking': g_looking,
                'c_looking': c_looking,
                'g_score': g_score,
                'c_score': c_score,
                'g_mem_frames': id_mgr.memories['GUARDIAN'].frames_visible,
                'k_mem_frames': id_mgr.memories['CHILD'].frames_visible,
                'recording': recording,
                'gpu_name': gpu_name_short,
                'vram_used': last_vram_mb,
                'assignment_uncertain': assignment_uncertain,
                'uncertainty_reason': (f'n_valid={n_valid}, need 2' if not exactly_two_valid else uncertainty_reason) or ('<2 confirmed people' if not both_present else '-'),
            })

            if not recording:
                if not exactly_two_valid:
                    reason = f"n_valid={n_valid}, need exactly 2"
                elif not both_present:
                    reason = "<2 confirmed people"
                elif assignment_uncertain:
                    reason = uncertainty_reason or "role uncertainty"
                else:
                    reason = "unknown"
                cv2.putText(frame, f"NO DATA ({reason})", (20, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

            if recording:
                frame_data = {
                    'frame_id': current_frame,
                    'timestamp_s': round(current_frame / fps, 3),
                    'id_locked': id_mgr.is_locked,
                    'mutual_attention': bool(g_looking and c_looking),
                    'people': []
                }
                for role in ['GUARDIAN', 'CHILD']:
                    if role in identified:
                        person = identified[role]
                        kpts = person['kpts']
                        pos = kpts[0][:2]
                        is_look = g_looking if role == 'GUARDIAN' else c_looking
                        sc = g_score if role == 'GUARDIAN' else c_score
                        co = g_conf if role == 'GUARDIAN' else c_conf

                        kpt_names = ['nose','left_eye','right_eye','left_ear','right_ear',
                                     'left_shoulder','right_shoulder','left_elbow','right_elbow',
                                     'left_wrist','right_wrist','left_hip','right_hip',
                                     'left_knee','right_knee','left_ankle','right_ankle']
                        kpts_export = [{
                            'name': name,
                            'x': round(float(kpts[ki][0]), 1),
                            'y': round(float(kpts[ki][1]), 1),
                            'conf': round(float(kpts[ki][2]), 3)
                        } for ki, name in enumerate(kpt_names)]

                        bbox_data = None
                        if 'bbox' in person and person['bbox'] is not None:
                            b = person['bbox']
                            bbox_data = [round(float(b[0])), round(float(b[1])), round(float(b[2])), round(float(b[3]))]

                        posture = {}
                        l_sh, r_sh = kpts[5], kpts[6]
                        l_hip, r_hip = kpts[11], kpts[12]
                        if l_sh[2] > 0.1 and r_sh[2] > 0.1:
                            posture['shoulder_tilt_degrees'] = round(float(np.degrees(np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0]))), 1)
                        if (l_sh[2] > 0.1 or r_sh[2] > 0.1) and (l_hip[2] > 0.1 or r_hip[2] > 0.1):
                            best_sh = l_sh if l_sh[2] > r_sh[2] else r_sh
                            best_hip = l_hip if l_hip[2] > r_hip[2] else r_hip
                            posture['torso_height_px'] = round(float(np.linalg.norm(best_sh[:2] - best_hip[:2])), 1)
                        l_wrist, r_wrist = kpts[9], kpts[10]
                        if l_wrist[2] > 0.1 and r_wrist[2] > 0.1:
                            posture['hand_distance_px'] = round(float(np.linalg.norm(l_wrist[:2] - r_wrist[:2])), 1)

                        role_fit = id_mgr.role_fit_score(person, role, frame=frame, current_frame=current_frame, use_memory=False)
                        frame_data['people'].append({
                            'role': role,
                            'track_id': int(person['track_id']),
                            'role_fit_score': round(float(role_fit), 4),
                            'is_looking': bool(is_look),
                            'looking_score': round(float(sc), 4),
                            'gaze_confidence': round(float(co), 4),
                            'detection_confidence': round(float(person.get('det_conf', 0.0)), 4),
                            'human_score': round(float(person.get('human_score', 0.0)), 4),
                            'pos_head': [round(float(pos[0]), 1), round(float(pos[1]), 1)],
                            'bbox': bbox_data,
                            'keypoints': kpts_export,
                            'posture': posture,
                        })

                if json_frames_written > 0:
                    json_file.write(',\n')
                json.dump(frame_data, json_file, indent=config.json_indent, ensure_ascii=False)
                json_frames_written += 1
                stats_counter['frames_with_data'] += 1
                if json_frames_written % max(1, config.json_flush_interval) == 0:
                    json_file.flush()
            else:
                stats_counter['frames_discarded'] += 1

            writer.write(frame)
            progress.update(1)

        except Exception as e:
            fatal_error = e
            print(f"\n[FATAL] Exception at frame {locals().get('current_frame', '?')}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break

    elapsed = time.time() - t_start
    for fn in [progress.close, cap.release, writer.release, cv2.destroyAllWindows]:
        try:
            fn()
        except Exception:
            pass

    try:
        if not json_file.closed:
            json_file.write('\n]')
            json_file.close()
    except Exception:
        pass
    print(f">>> JSON streaming: {json_frames_written} frames written.", flush=True)

    try:
        if not debug_log.closed:
            debug_log.flush()
            debug_log.close()
    except Exception:
        pass

    if fatal_error is not None:
        raise fatal_error

    if gpu_info['cuda_available']:
        peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_vram = 0

    del model
    import gc
    gc.collect()
    if gpu_info['cuda_available']:
        torch.cuda.empty_cache()

    total_proc = stats_counter['total_processed']
    with_data = stats_counter['frames_with_data']
    discarded = stats_counter['frames_discarded']
    g_look = stats_counter['g_looking_frames']
    c_look = stats_counter['c_looking_frames']
    mutual = stats_counter['mutual_attention_frames']
    reid_att = stats_counter['reid_attempts']
    reid_ok = stats_counter['reid_successes']
    multipass = stats_counter['multipass_used']
    hm_resets = stats_counter['honeymoon_resets']
    swap_events = stats_counter['swap_events']

    report = f"""
{'='*70}
  TITAN v58 — GAIA — FINAL REPORT
{'='*70}

  HARDWARE:
    GPU:          {gpu_info['device_name']}
    CUDA:         {gpu_info['cuda_version']}
    FP16:         {'YES' if use_half else 'NO'}
    VRAM peak:    {peak_vram} MB
    CPU cores:    {n_cores}
    Model:        {config.model_name} @ {config.imgsz}px

  v58 IMPROVEMENTS:
    Human validation:     SCORE-BASED (anti-toy / anti-false-human)
    Pair selection:       BEST-PAIR selection when 3+ candidates appear
    Role lock:            pair-based role-fit + relative order checks
    ReID / recovery:      role-fit + memory + track trust + bans
    Stale IDs cleanup:    YES (positive + negative + stale history)
    Pair consistency:     pair_score(current) vs pair_score(swapped)
    Adaptive lock:        {'YES (similar pair mode)' if id_mgr.similar_pair_mode else 'NO (distinct pair)'}
    SW reliable:          {'YES' if (prescan_profiles or {}).get('sw_reliable', True) else 'NO (weight reduced)'}
    Re-lock acceleration: {'YES (memories alive)' if any(id_mgr.memories[r].frames_visible > 30 for r in ['GUARDIAN', 'CHILD']) else 'NO'}
    Honeymoon resets:     {hm_resets}
    Role instability resets: {swap_events}

  PROCESSING:
    Frames processed:    {total_proc}
    Frames with data:    {with_data} ({100*with_data/max(total_proc,1):.1f}%)
    Frames discarded:    {discarded} ({100*discarded/max(total_proc,1):.1f}%)
    Total time:          {elapsed:.1f}s ({total_proc/max(elapsed,0.1):.1f} fps)

  DETECTION:
    Multi-pass used:     {multipass} times ({100*multipass/max(total_proc,1):.1f}% of frames)
    ReID attempts:       {reid_att}
    ReID recoveries:     {reid_ok} ({100*reid_ok/max(reid_att,1):.0f}%)

  ATTENTION:
    GUARDIAN → CHILD:    {g_look}/{with_data} ({100*g_look/max(with_data,1):.1f}%)
    CHILD → GUARDIAN:    {c_look}/{with_data} ({100*c_look/max(with_data,1):.1f}%)
    MUTUAL:              {mutual}/{with_data} ({100*mutual/max(with_data,1):.1f}%)

  OUTPUTS:
    Video:    {out_video}
    JSON:     {out_json}
    Report:   {out_report}
{'='*70}
"""
    print(report)
    with open(out_report, 'w', encoding='utf-8') as f:
        f.write(report)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 10: ENTRY POINT# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SEÇÃO 10: ENTRY POINT                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_check(config: TitanConfig):
    print("=" * 70)
    print("  TITAN v58 — CHECK MODE")
    print("=" * 70)
    gpu_info = diagnose_gpu()
    print_gpu_banner(gpu_info)
    print(f"\n>>> Video: {config.input_video}")
    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        print("  ✗ COULD NOT OPEN VIDEO"); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✓ {w}x{h} @ {fps:.1f}fps, {total} frames ({total/max(fps,1):.0f}s)")
    print(f"\n>>> Model: {config.model_name}")
    device   = 0 if gpu_info['cuda_available'] else 'cpu'
    model    = YOLO(config.model_name)
    use_half = config.use_fp16 and gpu_info['cuda_available'] and gpu_info.get('fp16_supported', False)
    ret, frame = cap.read()
    if not ret:
        print("  ✗ Could not read frame"); cap.release(); return
    print(f"  ✓ Frame read: {frame.shape}")
    results  = model.track(frame, persist=True, verbose=False, tracker="botsort.yaml",
                           classes=[0], conf=0.30, iou=0.65, half=use_half,
                           imgsz=config.imgsz, device=device)
    n_det     = len(results[0].boxes) if results[0].boxes is not None else 0
    n_tracked = len(results[0].boxes.id) if results[0].boxes.id is not None else 0
    print(f"  ✓ Detection: {n_det} boxes, {n_tracked} tracked")
    if MEDIAPIPE_AVAILABLE:
        gaze = GazeEstimator()
        gaze._init_mediapipe()
        n_lm = "478 (iris active)" if gaze._mp_available else "FAILED"
        print(f"  ✓ MediaPipe: {n_lm}")
    else:
        print("  ✗ MediaPipe: not available")
    if gpu_info['cuda_available']:
        t0 = time.time()
        for _ in range(5):
            model.track(frame, persist=True, verbose=False, tracker="botsort.yaml",
                        classes=[0], conf=0.30, iou=0.65, half=use_half,
                        imgsz=config.imgsz, device=device)
        avg_ms   = (time.time() - t0) / 5 * 1000
        est_fps  = 1000 / avg_ms
        est_hours = total / est_fps / 3600
        print(f"\n>>> Performance: ~{avg_ms:.0f}ms/frame → ~{est_fps:.1f}fps → ~{est_hours:.1f}h for {total} frames")
    cap.release()
    print(f"\n{'='*70}")
    print("  CHECK COMPLETE — all OK to run.")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys as _sys
    import glob as _glob

    _check_mode = '--check' in _sys.argv

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    processed_file = os.path.join(script_dir, "processed.txt")

    processed_set = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_set = {line.strip() for line in f if line.strip()}

    # ── v58: percorre Neurotipico E TEA ──
    groups = ["Neurotipico", "TEA"]
    cam_folders = ["cam2", "cam1"]

    video_queue = []  # (video_path, rel_path, group_name)
    for group in groups:
        input_base  = os.path.join(script_dir, "input",  group)
        if not os.path.isdir(input_base):
            print(f"[INFO] Input group not found: {input_base}, skipping.")
            continue
        for cam in cam_folders:
            cam_dir = os.path.join(input_base, cam)
            if not os.path.isdir(cam_dir):
                print(f"[WARN] Folder not found: {cam_dir}, skipping.")
                continue
            for mp4_path in sorted(_glob.glob(os.path.join(cam_dir, "*.mp4"))):
                rel_path = os.path.relpath(mp4_path, script_dir)
                if rel_path not in processed_set:
                    video_queue.append((mp4_path, rel_path, group))
                else:
                    print(f"  [SKIP] Already processed: {rel_path}")

    if not video_queue:
        print("=" * 70)
        print("  All videos already processed! Nothing to do.")
        print(f"  (Delete or edit '{processed_file}' to reprocess)")
        print("=" * 70)
        _sys.exit(0)

    print("=" * 70)
    print(f"  GAIA BATCH — {len(video_queue)} video(s) to process")
    print("=" * 70)
    for i, (vpath, vrel, grp) in enumerate(video_queue, 1):
        print(f"  {i}. [{grp}] {vrel}")
    print("=" * 70)

    for vid_idx, (video_path, rel_path, group) in enumerate(video_queue, 1):
        output_base  = os.path.join(script_dir, "output", group)
        video_name   = os.path.splitext(os.path.basename(video_path))[0]
        out_vid_path = os.path.join(output_base, "video",    f"outputvideo_{video_name}.mp4")
        out_json_path= os.path.join(output_base, "json",     f"json_{video_name}.json")
        out_rpt_path = os.path.join(output_base, "relatorio",f"report_{video_name}.txt")

        print(f"\n{'='*70}")
        print(f"  [{vid_idx}/{len(video_queue)}] [{group}] Processing: {rel_path}")
        print(f"{'='*70}\n")

        config = TitanConfig(
            input_video   = video_path,
            output_video  = os.path.relpath(out_vid_path,  script_dir),
            output_json   = os.path.relpath(out_json_path, script_dir),
            output_report = os.path.relpath(out_rpt_path,  script_dir),
        )

        try:
            if _check_mode:
                run_check(config); break
            else:
                run_gaia(config)
                with open(processed_file, 'a', encoding='utf-8') as f:
                    f.write(rel_path + '\n')
                print(f"\n>>> Marked as processed: {rel_path}")

        except KeyboardInterrupt:
            print(f"\n  INTERRUPTED. '{video_name}' NOT marked as processed.")
            _sys.exit(1)

        except Exception as e:
            print(f"\n  FATAL ERROR on '{video_name}': {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

        finally:
            import gc; gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    print(f"\n{'='*70}")
    print(f"  GAIA BATCH COMPLETE — {len(video_queue)} video(s) done")
    print(f"{'='*70}")