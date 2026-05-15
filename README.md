<p align="center">
  <img src="assets/gaia.gif" alt="Logo do Projeto GAIA" width="500"/>
</p>
<h1 align="center">GAIA — Gaze and Action Interaction Analyzer</h1>
<p align="center">
  <em>Pipeline de Visão Computacional + LSTM para Análise Comportamental Automatizada como Ferramenta Auxiliar de Prognóstico do TEA</em>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Status-Concluído-brightgreen" alt="Status do Projeto">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Versão do Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Pipeline-TITAN%20v58-blueviolet" alt="TITAN v58">
  <img src="https://img.shields.io/badge/Classificador-GAIA%20LSTM%20v1.0-orange" alt="GAIA LSTM">
  <img src="https://img.shields.io/badge/Acurácia-83.3%25-success" alt="Acurácia">
  <img src="https://img.shields.io/badge/AUC--ROC-0.759-success" alt="AUC-ROC">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="Licença">
</p>

---

## 📖 Sobre o Projeto

O diagnóstico do Transtorno do Espectro Autista (TEA) é um processo longo, subjetivo e caro — com tempo médio de espera de 3+ anos e dependência de equipamentos como eye-trackers profissionais que custam milhares de reais.

O **GAIA** é um sistema de visão computacional que analisa automaticamente vídeos de interação entre guardiões e crianças usando **câmeras convencionais**, extraindo métricas objetivas de atenção visual, proximidade interpessoal e linguagem corporal. Essas métricas alimentam um classificador **Bi-LSTM com atenção temporal** que produz um indicador probabilístico de risco P(TEA).

> **Trabalho de Conclusão de Curso** — Ciência da Computação, Centro Universitário FEI (2026)

---

## 🏗️ Arquitetura

O sistema é composto por dois módulos principais:

### TITAN v58 — Pipeline de Extração

```
Vídeo → YOLOv8l-pose → BoTSORT → MediaPipe Face Mesh → JSON frame-a-frame
              │              │              │
        17 keypoints    Rastreamento    Íris landmarks
        por pessoa      de identidade   → direção do olhar
```

- **YOLOv8l-pose**: Detecção de esqueleto (17 keypoints COCO) em tempo real
- **BoTSORT**: Rastreamento multi-objeto com re-identificação
- **MediaPipe Face Mesh**: 478 landmarks faciais + íris para estimativa de gaze
- **Classificação de papéis**: Identifica guardião vs. criança por tamanho relativo
- **Output**: JSON com métricas frame-a-frame (gaze scores, postura, proximidade)

### GAIA LSTM v1.0 — Classificador Temporal

```
JSON → Feature Extractor (14 features) → Sliding Windows (10s) → Bi-LSTM → P(TEA)
```

- **14 features** em 4 dimensões comportamentais:
  - 🔍 **Gaze**: scores de atenção G→C e C→G, atenção mútua, assimetria
  - 📏 **Proximidade**: distância cabeça-a-cabeça, velocidade de variação
  - 🏃 **Postura/Motor**: inclinação de ombros, distância entre mãos, deslocamento da cabeça
  - ⏱️ **Temporal**: duração de episódios contínuos de atenção
- **Bi-LSTM**: 2 camadas, 128 unidades ocultas, bidirecional
- **Temporal Attention Pooling**: aprende quais janelas de 10s são mais discriminativas
- **Validação**: Leave-One-Subject-Out (LOSO) por participante — 7 folds

---

## 📊 Resultados

### Dataset

| | NT | TEA | Total |
|---|---|---|---|
| Participantes | 5 | 5 | 10 |
| Gravações (2 câmeras/participante) | 10 | 10 | 20 |
| Quadros processados | 485.963 | 1.191.738 | **1.677.701** |
| Taxa de frames válidos | 62,0% | 30,9% | — |

**Fonte**: PCRC (Parent-Child Research Clinic), University of New South Wales (UNSW), Austrália.

### Achados Descritivos (melhor câmera por participante)

| Métrica | NT (n=4) | TEA (n=3) | Δ |
|---|---|---|---|
| Criança → Guardião | 60,5% | 47,4% | **13,1 pp** |
| Guardião → Criança | 53,9% | 66,9% | Guardião TEA compensa |
| Distância quadril | 292 px | 421 px | **+44% (IC 95% sem overlap)** |
| Mãos da criança | 68 px | 88 px | **+29% (IC 95% sem overlap)** |
| Variabilidade motora (σ) | 0,7 | 3,1 | **4,4× maior** |

### Classificador LSTM

| Métrica | Valor |
|---|---|
| **Acurácia por sessão** | **83,3% (10/12)** |
| Acurácia por janela | 68,7% |
| Sensibilidade (TEA) | 51,9% |
| Especificidade (NT) | 82,9% |
| Precisão | 72,1% |
| F1-Score | 60,4% |
| **AUC-ROC** | **0,759** |

- ✅ NT: **8/8 corretos** (P(TEA) range: 0,11 – 0,46)
- ✅ TEA: 2/4 corretos. Erros borderline: P(TEA) = 0,487 e 0,495

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Detecção de pose | YOLOv8l-pose (Ultralytics) |
| Rastreamento | BoTSORT |
| Estimativa de gaze | MediaPipe Face Mesh + Iris |
| Classificador | PyTorch (Bi-LSTM + Attention) |
| Análise de dados | NumPy, SciPy |
| Relatórios | ReportLab (PDF) |
| Interface | Streamlit |
| Hardware | RTX 5060, Ryzen 7 5700X, 32GB RAM |

---

## 🚀 Como Usar

### Requisitos

- Python 3.9+
- CUDA 12.x (recomendado para GPU)
- ~4GB de VRAM

### Instalação

```bash
git clone https://github.com/gabrielbalbine/gaia.git
cd gaia
pip install -r requirements.txt
```

### Pipeline de Extração (TITAN)

```bash
# Processar um vídeo e gerar JSON de métricas
python titan_pipeline.py --input video.mp4 --output output/
```

### Treinamento LSTM

```bash
# Treinar com LOSO cross-validation (pipeline completa)
python gaia_lstm_v1.py

# Customizar hiperparâmetros
python gaia_lstm_v1.py --hidden 256 --layers 3 --dropout 0.4 --window 450
```

### Inferência em nova sessão

```bash
python gaia_lstm_v1.py --predict path/to/session.json --model lstm_output/models/gaia_lstm_final.pt
```

---

## 📁 Estrutura do Projeto

```
gaia/
├── titan_pipeline.py          # Pipeline de extração (TITAN v58)
├── gaia_lstm_v1.py            # Classificador LSTM + treinamento LOSO
├── posture_analysis.py        # Análise de postura standalone
├── output/
│   ├── Neurotipico/json/      # JSONs frame-a-frame (NT)
│   └── TEA/json/              # JSONs frame-a-frame (TEA)
├── lstm_output/
│   ├── models/                # Modelo treinado (.pt)
│   ├── reports/               # Relatórios descritivos e prognósticos
│   └── results/               # loso_results.json
├── assets/                    # Logo e recursos visuais
└── requirements.txt
```

---

## ⚠️ Nota Ética

O GAIA é uma **ferramenta auxiliar de prognóstico**. Os resultados **não constituem diagnóstico clínico** e devem ser interpretados exclusivamente por profissionais qualificados (neuropsicólogos, neuropediatras, psiquiatras infantis).

---

## ✍️ Autor

**Gabriel Balbine de Andrades**  
Ciência da Computação — Centro Universitário FEI

### Orientador

**Prof. Dr. Victor Perrone de Lima Varela**  
Departamento de Ciência da Computação, Centro Universitário FEI

### Colaboração

**PCRC — Parent-Child Research Clinic**  
University of New South Wales (UNSW), Austrália

---

## 📄 Licença

Este projeto está sob a licença MIT.
