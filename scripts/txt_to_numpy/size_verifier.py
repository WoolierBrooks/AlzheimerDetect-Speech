import os
import numpy as np


pasta = "<diretório da pasta>"

# Inicialização das variáveis para armazenar o tamanho da menor e maior lista
tamanho_menor_lista = float('inf')
tamanho_maior_lista = 0

# Percorre todos os arquivos na pasta
for arquivo in os.listdir(pasta):
    if arquivo.endswith('.npy'):
        # Carrega o arquivo npy
        caminho_arquivo = os.path.join(pasta, arquivo)
        lista = np.load(caminho_arquivo)

        # Verifica o tamanho da lista
        tamanho_lista = len(lista)

        # Atualiza as variáveis de menor e maior tamanho, se necessário
        if tamanho_lista < tamanho_menor_lista:
            tamanho_menor_lista = tamanho_lista
        if tamanho_lista > tamanho_maior_lista:
            tamanho_maior_lista = tamanho_lista

# Resultados
print(f"Tamanho da menor lista: {tamanho_menor_lista}")
print(f"Tamanho da maior lista: {tamanho_maior_lista}")