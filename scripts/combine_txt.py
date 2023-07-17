import os

def combine_files(input_folder, output_folder):
    # Criar um dicionário para armazenar os arrays combinados pelo nome base
    combined_arrays = {}

    # Ler arquivos txt e combinar arrays por nome base
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r") as file:
                base_name = "-".join(filename.split("-")[:-1]) + ".txt"
                data = file.read()
                if base_name in combined_arrays:
                    combined_arrays[base_name].append(data)
                else:
                    combined_arrays[base_name] = [data]

    # Criar a pasta de saída se ela não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Escrever os arrays combinados nos arquivos de saída
    for base_name, arrays in combined_arrays.items():
        output_filepath = os.path.join(output_folder, base_name)
        with open(output_filepath, "w") as outfile:
            for array in arrays:
                outfile.write(array.strip())  # Elimina os espaços em branco à esquerda e à direita
                outfile.write("\n")  # Adiciona uma linha em branco entre os arrays

input_folder = "<diretório de entrada>"
output_folder = "<diretório de saída>"

combine_files(input_folder, output_folder)