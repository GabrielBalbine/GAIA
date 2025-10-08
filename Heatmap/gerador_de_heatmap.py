# analisador_de_atencao_final.py (VERSÃO 11 - LÓGICA DUALISTA E ESTÁVEL)

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import json

# --- CONFIGURAÇÃO ---
MODELO_YOLO = 'yolov8m-pose.pt'
VIDEO_ENTRADA = "video_teste_2_pessoas.mp4" 

# Nomes dos arquivos de saída
VIDEO_SAIDA_ANALISE = "video_analisado_com_vetores.mp4"
HEATMAP_SAIDA_CRIANCA = "heatmap_atencao_crianca.png"
HEATMAP_SAIDA_GUARDIAO = "heatmap_atencao_guardiao.png"
JSON_SAIDA = "dados_completos_visao.json"

# --- FUNÇÕES AUXILIARES ---

def calcular_altura_tronco(keypoints):
    ombro_esq = keypoints.get(5); ombro_dir = keypoints.get(6)
    quadril_esq = keypoints.get(11); quadril_dir = keypoints.get(12)
    if not all([ombro_esq, ombro_dir, quadril_esq, quadril_dir]): return 0
    ponto_medio_ombros = ((ombro_esq[0] + ombro_dir[0]) / 2, (ombro_esq[1] + ombro_dir[1]) / 2)
    ponto_medio_quadril = ((quadril_esq[0] + quadril_dir[0]) / 2, (quadril_esq[1] + quadril_dir[1]) / 2)
    return np.sqrt((ponto_medio_ombros[0] - ponto_medio_quadril[0])**2 + (ponto_medio_ombros[1] - ponto_medio_quadril[1])**2)

def verifica_olhar_para_alvo(kpts_observador, kpts_alvo):
    nariz_obs = kpts_observador.get(0); orelha_esq_obs = kpts_observador.get(3); orelha_dir_obs = kpts_observador.get(4)
    pontos_alvo = [kpts_alvo.get(i) for i in [0, 5, 6]]; pontos_alvo_validos = [pt for pt in pontos_alvo if pt is not None]
    if not pontos_alvo_validos: return False, None
    ponto_alvo = (np.mean([pt[0] for pt in pontos_alvo_validos]), np.mean([pt[1] for pt in pontos_alvo_validos]))
    ponto_traseiro = None
    if orelha_esq_obs and orelha_dir_obs:
        ponto_traseiro = (np.mean([orelha_esq_obs[0], orelha_dir_obs[0]]), np.mean([orelha_esq_obs[1], orelha_dir_obs[1]]))
    if not ponto_traseiro or not nariz_obs: return False, None
    vetor_cabeca = np.array([nariz_obs[0] - ponto_traseiro[0], nariz_obs[1] - ponto_traseiro[1]])
    vetor_para_alvo = np.array([ponto_alvo[0] - nariz_obs[0], ponto_alvo[1] - nariz_obs[1]])
    norma_cabeca = np.linalg.norm(vetor_cabeca); norma_alvo = np.linalg.norm(vetor_para_alvo)
    if norma_cabeca == 0 or norma_alvo == 0: return False, None
    vetor_cabeca_norm = vetor_cabeca / norma_cabeca; vetor_para_alvo_norm = vetor_para_alvo / norma_alvo
    produto_escalar = np.dot(vetor_cabeca_norm, vetor_para_alvo_norm)
    
    # Critério super rigoroso: cosseno > 0.94 (~20 graus de cone de visão)
    return produto_escalar > 0.94, ponto_alvo

def get_ponto_foco_ambiente(kpts_observador):
    nariz = kpts_observador.get(0); olho_esq = kpts_observador.get(1); olho_dir = kpts_observador.get(2)
    if not all([nariz, olho_esq, olho_dir]): return None, None
    ponto_medio_olhos = (int((olho_esq[0] + olho_dir[0]) / 2), int((olho_esq[1] + olho_dir[1]) / 2))
    vetor_bruto = (nariz[0] - ponto_medio_olhos[0], nariz[1] - ponto_medio_olhos[1])
    vetor_amortecido = (vetor_bruto[0], max(0, vetor_bruto[1])) # Garante que o vetor nunca aponte para cima
    norma = np.linalg.norm(vetor_amortecido)
    if norma == 0: return None, None
    vetor_unitario = vetor_amortecido / norma
    distancia_projecao = 300 # Distância fixa para projetar no "chão"
    ponto_foco = (int(ponto_medio_olhos[0] + vetor_unitario[0] * distancia_projecao), int(ponto_medio_olhos[1] + vetor_unitario[1] * distancia_projecao))
    return ponto_medio_olhos, ponto_foco

# --- BLOCO PRINCIPAL ---
def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    video_path = os.path.join(script_dir, VIDEO_ENTRADA)
    if not os.path.exists(video_path): print(f"ERRO: Vídeo não encontrado: '{video_path}'"); sys.exit(1)
    print(">>> Carregando modelo de IA..."); model = YOLO(MODELO_YOLO)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, first_frame = cap.read()
    if not ret: print("ERRO: Não foi possível ler o vídeo."); cap.release(); sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(script_dir, VIDEO_SAIDA_ANALISE), fourcc, fps, (frame_width, frame_height))
    heatmap_crianca = np.zeros((frame_height, frame_width), dtype=np.float32)
    heatmap_guardiao = np.zeros((frame_height, frame_width), dtype=np.float32)
    dados_finais_json = {'metadata': {}, 'frames': []}
    identidades, memoria_tamanho, frame_count = {}, {}, 0
    print(">>> Analisando vídeo com lógica DUALISTA e ESTÁVEL...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        print(f"Processando frame: {frame_count}", end='\r')
        dados_do_frame_atual = {'frame_id': frame_count, 'analisado': False, 'motivo': '', 'pessoas': []}
        results = model.track(frame, persist=True, verbose=False)
        ids_detectados = results[0].boxes.id
        if ids_detectados is not None and len(ids_detectados) == 2:
            track_ids = ids_detectados.int().cpu().tolist()
            keypoints_por_id = {track_id: {idx: tuple(map(int, pt)) for idx, pt in enumerate(results[0].keypoints.xy[i].cpu().numpy())} for i, track_id in enumerate(track_ids)}
            for track_id, keypoints in keypoints_por_id.items():
                altura_tronco = calcular_altura_tronco(keypoints)
                if altura_tronco > 0: memoria_tamanho[track_id] = altura_tronco
            if len(identidades) < 2 and len(memoria_tamanho) == 2:
                id1, id2 = memoria_tamanho.keys()
                identidades[id1] = 'Guardiao' if memoria_tamanho[id1] > memoria_tamanho[id2] else 'Crianca'
                identidades[id2] = 'Crianca' if memoria_tamanho[id1] > memoria_tamanho[id2] else 'Guardiao'
            id_crianca, id_guardiao = None, None
            for track_id, papel in identidades.items():
                if papel == 'Crianca': id_crianca = track_id
                else: id_guardiao = track_id
            if id_crianca is not None and id_guardiao is not None:
                kpts_crianca, kpts_guardiao = keypoints_por_id.get(id_crianca), keypoints_por_id.get(id_guardiao)
                if kpts_crianca and kpts_guardiao:
                    dados_do_frame_atual['analisado'] = True
                    for id_obs, kpts_obs, kpts_alvo, heatmap in [(id_crianca, kpts_crianca, kpts_guardiao, heatmap_crianca), (id_guardiao, kpts_guardiao, kpts_crianca, heatmap_guardiao)]:
                        papel = identidades.get(id_obs)
                        cor = (0, 0, 255) if papel == 'Crianca' else (255, 0, 0)
                        cv2.putText(frame, papel, (kpts_obs.get(0, (50,50))[0], kpts_obs.get(0, (50,50))[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
                        
                        olhando_para_parceiro, ponto_alvo_central = verifica_olhar_para_alvo(kpts_obs, kpts_alvo)
                        ponto_foco_final = None

                        if olhando_para_parceiro:
                            ponto_origem = kpts_obs.get(0)
                            if ponto_origem and ponto_alvo_central:
                                cv2.line(frame, ponto_origem, (int(ponto_alvo_central[0]), int(ponto_alvo_central[1])), cor, 2)
                                cv2.circle(frame, (int(ponto_alvo_central[0]), int(ponto_alvo_central[1])), 10, cor, 2)
                                ponto_foco_final = ponto_alvo_central
                        else:
                            ponto_origem, ponto_foco_ambiente = get_ponto_foco_ambiente(kpts_obs)
                            if ponto_origem and ponto_foco_ambiente:
                                cv2.line(frame, ponto_origem, ponto_foco_ambiente, cor, 1)
                                ponto_foco_final = ponto_foco_ambiente
                        
                        if ponto_foco_final:
                            x, y = int(ponto_foco_final[0]), int(ponto_foco_final[1])
                            if 0 <= x < frame_width and 0 <= y < frame_height:
                                cv2.circle(heatmap, (x, y), radius=40, color=1, thickness=-1)
                        
                        dados_do_frame_atual['pessoas'].append({'track_id': id_obs, 'papel': papel, 'olhando_para_parceiro': bool(olhando_para_parceiro)})
        else:
            motivo = "Numero de pessoas invalido"
            dados_do_frame_atual['motivo'] = motivo; cv2.putText(frame, motivo, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        dados_finais_json['frames'].append(dados_do_frame_atual)
        out_video.write(frame)
    print("\n>>> Processamento de vídeo concluído..."); cap.release(); out_video.release()
    with open(os.path.join(script_dir, JSON_SAIDA), 'w') as f: json.dump(dados_finais_json, f, indent=4)
    for heatmap_data, papel, output_path in [(heatmap_crianca, "Crianca", HEATMAP_SAIDA_CRIANCA), (heatmap_guardiao, "Guardiao", HEATMAP_SAIDA_GUARDIAO)]:
        heatmap_data = cv2.GaussianBlur(heatmap_data, (25, 25), 0)
        if np.max(heatmap_data) > 0: heatmap_data = (heatmap_data / np.max(heatmap_data) * 255)
        heatmap_uint8 = heatmap_data.astype(np.uint8)
        heatmap_colorido = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        imagem_final = cv2.addWeighted(first_frame, 0.5, heatmap_colorido, 0.5, 0)
        cv2.putText(imagem_final, f"Mapa de Calor - Foco do(a) {papel}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(script_dir, output_path), imagem_final)
    cv2.destroyAllWindows()
    print(">>> TUDO PRONTO!")

if __name__ == "__main__":
    main()