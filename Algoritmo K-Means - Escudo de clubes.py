import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Funções para Extração de Características ---
def extrair_cor_media(imagem):
    """Calcula a cor média (R, G, B) de uma imagem."""
    # O cv2 lê em BGR, então a média também será BGR. Invertemos para RGB no final.
    cor_media_bgr = np.mean(imagem, axis=(0, 1))
    return cor_media_bgr[::-1]

def extrair_momentos_hu(imagem):
    """Extrai os 7 Momentos de Hu, que são invariantes a escala, rotação e translação."""
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Binariza a imagem para encontrar os contornos. Assumimos que o escudo não é totalmente preto.
    _, imagem_binaria = cv2.threshold(imagem_cinza, 1, 255, cv2.THRESH_BINARY)
    momentos = cv2.moments(imagem_binaria)
    momentos_hu = cv2.HuMoments(momentos)

    # Aplicar log para estabilizar a escala dos valores, pois eles variam muito
    # Adicionamos um valor pequeno para evitar log(0)
    return -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu) + 1e-7)

# --- Carregamento e Processamento das Imagens ---
caminho_da_pasta = '/content/Escudo dos clubes' # Criar pasta Escudo dos clubes no seu Colab
arquivos_imagem = [f for f in os.listdir(caminho_da_pasta) if f.endswith(('.png', '.jpg', '.jpeg'))]
lista_de_caracteristicas = []

for nome_arquivo in arquivos_imagem:
    caminho_completo = os.path.join(caminho_da_pasta, nome_arquivo)
    imagem = cv2.imread(caminho_completo)

    if imagem is not None:
        cor = extrair_cor_media(imagem)
        forma = extrair_momentos_hu(imagem).flatten() # Aplainar para garantir que seja 1D

        # Combinar características de cor e forma em um único vetor
        caracteristicas_combinadas = np.concatenate([cor, forma])
        lista_de_caracteristicas.append(caracteristicas_combinadas)
X = np.array(lista_de_caracteristicas)

# --- Padronização e Clusterização ---
# Padronizar as características para que tenham a mesma escala
scaler = StandardScaler()
caracteristicas_escaladas = scaler.fit_transform(X)

inercias = []
range_k = range(1, 11)

print("Calculando a inércia para cada valor de k (1 a 10)...")

for k in range_k:
    # 1. Cria e treina o modelo KMeans para o valor 'k' atual
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(caracteristicas_escaladas)

    # 2. Armazena a inércia do modelo treinado
    # Inércia = Soma das distâncias quadradas das amostras ao centro do cluster mais próximo (WCSS)
    inercias.append(kmeans.inertia_)
print("Cálculo finalizado. Gerando o gráfico...")

# 3. Plota o gráfico do Método do Cotovelo
plt.figure(figsize=(10, 6))
plt.plot(range_k, inercias, marker='o', linestyle='--')
plt.title('Método do Cotovelo para Escolha do Número de Clusters (k)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (WCSS - Within-Cluster Sum of Squares)')
plt.xticks(range_k) # Garante que o eixo X mostre todos os números de 1 a 10
plt.grid(True)
plt.show()