import os
import json

# CAMINHO DA PASTA COREL (Ajuste se necessário)
dataset_root = "./corel"
output_file = os.path.join(dataset_root, "metadata.jsonl")

MAPA_CLASSES = {
    "0001": "royal guards",
    "0002": "train",
    "0003": "dessert",
    "0004": "vegetables",
    "0005": "snow",
    "0006": "sunset",
}


def generate_metadata():
    print(f"Gerando metadados em: {output_file}")

    # Verifica se a pasta existe
    if not os.path.exists(dataset_root):
        print(f"ERRO: A pasta '{dataset_root}' não foi encontrada.")
        return

    count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        # Lista todos os arquivos da pasta
        arquivos = sorted(os.listdir(dataset_root))

        for filename in arquivos:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                class_id = filename.split('_')[0]

                class_name = MAPA_CLASSES[class_id]

                # Cria o prompt
                prompt = f"a photo of a {class_name}"

                # O huggingface datasets espera o caminho relativo à pasta do metadata
                data = {
                    "file_name": filename,
                    "text": prompt
                }

                f.write(json.dumps(data) + "\n")
                count += 1

    print(f"Concluído! {count} imagens processadas.")
    print(f"Arquivo salvo em: {output_file}")


if __name__ == "__main__":
    generate_metadata()
