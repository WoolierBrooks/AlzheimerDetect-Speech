import os
import numpy as np

def limpar_arquivo_txt(caminho_arquivo):
    # Ler o conteúdo do arquivo de texto
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()

    # Remover colchetes, espaços em branco e quebras de linha indesejados
    dados_limpos = []
    for linha in linhas:
        linha_limpa = linha.replace('[', '').replace(']', '').replace('\n', ' ').replace (',', '')#.replace(' ', '').rstrip(',')
        dados_limpos.append(linha_limpa)

    # Salvar os dados limpos em um novo arquivo
    caminho_arquivo_limpo = caminho_arquivo.replace('.txt', '_limpo.txt')
    with open(caminho_arquivo_limpo, 'w') as arquivo_limpo:
        for linha in dados_limpos:
            arquivo_limpo.write(linha)#.write(linha + '\n')

    return caminho_arquivo_limpo

# Caminho para a pasta que contém os arquivos de texto originais
pasta_origem = "<diretório de entrada>"

# Listar todos os arquivos de texto na pasta
arquivos_txt = [arquivo for arquivo in os.listdir(pasta_origem) if arquivo.endswith('.txt')]

# Processar cada arquivo de texto e salvar os arquivos limpos em uma nova pasta
pasta_destino = "<diretório de saída>"
os.makedirs(pasta_destino, exist_ok=True)

for arquivo_txt in arquivos_txt:
    caminho_arquivo_original = os.path.join(pasta_origem, arquivo_txt)
    caminho_arquivo_limpo = limpar_arquivo_txt(caminho_arquivo_original)
    caminho_arquivo_destino = os.path.join(pasta_destino, os.path.basename(caminho_arquivo_limpo))
    os.rename(caminho_arquivo_limpo, caminho_arquivo_destino)