# MELOBIRD
Um algoritmo para identificação de aves fundamentado em Python  
![Logo do projeto](https://github.com/Luiz-Przygoda/MeloBird/blob/main/Media/Melobird%20-%20logo%202.jpg)

## Introdução
A classificação automática de cantos de pássaros é um desafio relevante porque essas espécies reagem rapidamente às mudanças climáticas, tornando seu monitoramento essencial. No entanto, identificar manualmente cada canto é um processo lento, caro e sujeito a erros humanos. Este trabalho busca automatizar essa tarefa criando um sistema capaz de reconhecer espécies por meio de seus sons. Para isso, replicamos metodologias do artigo original, explorando diferentes arquiteturas de deep learning e técnicas de extração de características. O objetivo final é oferecer uma solução eficiente, escalável e precisa para auxiliar pesquisas ambientais e monitoramento da biodiversidade.

## Funcionalidades Principais
O sistema desenvolvido permite classificar automaticamente cantos de pássaros a partir de arquivos de áudio, utilizando um modelo de deep learning treinado especificamente para essa tarefa.  
Ele processa o áudio, extrai suas características acústicas e aplica o modelo para prever a espécie correspondente.  
A aplicação também disponibiliza uma interface interativa em Streamlit, onde o usuário pode enviar novos áudios e visualizar instantaneamente o resultado da classificação.  
Além disso, todo o pipeline, do pré-processamento ao modelo final, foi otimizado para garantir rapidez, precisão e facilidade de uso.  

## Tecnologias Utilizadas
- **Python**
- **Librosa** — Processamento de áudio e extração de características.
- **TensorFlow** / Keras — Carregamento e execução do modelo de deep learning.
- **Streamlit** — Interface web interativa.
- **Matplotlib** — Geração e visualização do espectrograma.
- **OpenCV (cv2)** — Leitura e exibição das imagens das espécies.
  
## Como Executar

Siga os passos abaixo para executar o projeto em sua máquina local.

### Pré-requisitos

* Python 3.8 ou superior instalado.
* `pip` (gerenciador de pacotes do Python) disponível no seu terminal.

### Passos de Instalação
1.  **Clone o repositório (ou baixe os arquivos):**
    ```bash
    git clone https://github.com/Luiz-Przygoda/MeloBird
    cd MeloBird
    ```

2.  **Instale as dependências necessárias:**
    ```bash
    pip install tensorflow
    pip install scikit-learn
    pip install opencv-python
    pip install librosa
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install streamlit
    pip install streamlit-extras
    pip install tqdm
    ```

3.  **Execute o programa:**
    ```bash
    python app.py
    ```


## **Colaboradores**
| [<img src="https://avatars.githubusercontent.com/u/142179999?v=4" width="115">](https://github.com/Luiz-Przygoda) | [<img src="https://avatars.githubusercontent.com/u/75136675?v=4" width="115">](https://github.com/marcobgh) | [<img src="https://avatars.githubusercontent.com/u/125486974?v=4" width="115">](https://github.com/mariaglx) | [<img src="https://avatars.githubusercontent.com/u/113839563?v=4" width="115">](https://github.com/Wyllye) |
|:--------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| **Luiz-Przygoda**                                                              | **Marcobgh**                                                               | **Mariaglx**                                                           | **Wyllye**                                                              |
