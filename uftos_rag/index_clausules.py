from datasets import load_dataset

dataset_info = load_dataset("lex_glue", "unfair_tos", split="train", streaming=True) # Usamos streaming para obtener info rápido
label_feature = dataset_info.features['labels']

if hasattr(label_feature, 'feature') and hasattr(label_feature.feature, 'names'):
    label_names = label_feature.feature.names
    print("Mapeo de Índice a Nombre de Etiqueta para UNFAIR-ToS:")
    for i, name in enumerate(label_names):
        print(f"Índice {i}: {name}")