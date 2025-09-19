<p align="center">
  <img src="assets/gaia.gif" alt="Logo do Projeto GAIA" width="500"/>
</p>

<h1 align="center">Projeto GAIA</h1>

<p align="center">
  <em>Uma Ferramenta de Visão Computacional para Análise Quantitativa de Padrões Comportamentais no Neurodesenvolvimento</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow" alt="Status do Projeto">
  <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Versão do Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="Licença">
</p>

---

## 📖 Sobre o Projeto

O diagnóstico de condições do neurodesenvolvimento, como o Transtorno do Espectro Autista (TEA), é um processo complexo e predominantemente clínico, que depende de observações comportamentais. O **GAIA** (acrônimo para *Gaze and Action Interaction Analyzer*) nasce como um projeto de TCC com o objetivo de criar uma ferramenta de suporte que traga mais **objetividade e dados quantitativos** para essa análise.

Utilizando visão computacional, o GAIA processa vídeos de interações (como entre pais e filhos) para extrair métricas automáticas de comportamento. A análise é focada em duas frentes principais:

1.  **Atenção Visual:** Onde a criança está olhando? Quanto tempo ela passa fixada em rostos em comparação com objetos? Para isso, o projeto implementará a detecção de **Áreas de Interesse (AOIs)**, mapeando o foco visual ao longo do tempo.
2.  **Movimento Corporal:** Como a criança se move e gesticula? O projeto utiliza **captura de pose (*pose estimation*)** para extrair o esqueleto corporal e permitir a futura análise de padrões motores.

O objetivo final é criar uma plataforma que possa ajudar pesquisadores e profissionais da saúde a identificar padrões comportamentais sutis em diferentes grupos (como crianças com TEA, neurotípicas e com traços de insensibilidade emocional), contribuindo para um entendimento mais profundo e auxiliando em intervenções futuras.

### ✨ Principais Funcionalidades (Atuais e Planejadas)

* ✅ **Captura de Pose (*Pose Estimation*):** Detecção dos 33 pontos-chave do corpo e geração de um vídeo com o esqueleto sobreposto. *(Etapa atual concluída)*
* ⏳ **Análise de Atenção Visual:** Detecção de rostos e definição de Áreas de Interesse (AOIs) para rastrear o olhar. *(Próxima etapa)*
* ⏳ **Extração de Métricas:** Calcular dados como tempo de fixação em AOIs, frequência de gestos, etc.
* ⏳ **Visualização de Dados:** Gerar relatórios e gráficos com as métricas extraídas.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.9**
* **OpenCV:** Para manipulação de vídeo.
* **MediaPipe:** Para o modelo de IA de detecção de pose e, futuramente, de faces e landmarks faciais.
* **Docker:** Para a criação de um ambiente de desenvolvimento 100% consistente e reprodutível.

---

## 🚀 Começando

Para executar este projeto, você precisará ter o [Docker](https://www.docker.com/products/docker-desktop/) instalado.

### Instalação

1.  **Clone o repositório:** `git clone https://github.com/seu-usuario/seu-repositorio.git`
2.  **Navegue até a pasta:** `cd seu-repositorio`
3.  **Construa e inicie o contêiner Docker:**
    ```sh
    docker-compose up -d --build
    ```

---

##  Como Usar (Módulo de Captura de Pose)

1.  **Prepare o vídeo:** Coloque um arquivo de vídeo (ex: `video_teste.mp4`) na raiz do projeto.
2.  **Ajuste o script:** Abra `captura_esqueleto.py` e altere a variável `input_video_path` para o nome do seu vídeo.
3.  **Execute a análise:**
    ```sh
    docker-compose exec esqueleto-tracker python captura_esqueleto.py
    ```
4.  **Baixe o resultado:** O vídeo processado (`esqueleto_output.mp4`) aparecerá na pasta, pronto para ser baixado.

---

## ✍️ Autores

* **Gabriel Balbine de Andrades**
* **Luiggi Paschoalini Garcia**

### Orientador

* **Prof. Dr. Victor Varela**

---

## 📄 Licença

Este projeto está sob a licença MIT.
