import os
import numpy as np

pasta_contendo_npy = "<diretório da pasta>"
tamanho_desejado = 287

arquivos_npy = [f for f in os.listdir(pasta_contendo_npy) if f.endswith(".npy")]

for nome_arquivo in arquivos_npy:
    caminho_arquivo = os.path.join(pasta_contendo_npy, nome_arquivo)
    array_original = np.load(caminho_arquivo, allow_pickle=True)
    
    if len(array_original.shape) == 2:  # Verifica se é um array 2D (matriz)
        array_cortado = array_original[:, :tamanho_desejado]
    else:
        array_cortado = array_original[:tamanho_desejado]
    
    novo_nome_arquivo = "cortado_" + nome_arquivo
    novo_caminho_arquivo = os.path.join(pasta_contendo_npy, novo_nome_arquivo)
    np.save(novo_caminho_arquivo, array_cortado)
