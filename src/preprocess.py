import os
import argparse
import torch
import yaml
import numpy as np
import requests
import gzip

def preprocess(config_path):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    root_dir = params['preprocess']['root_dir']
    raw_dir = os.path.join(root_dir, 'MNIST', 'raw')
    processed_dir = os.path.join(root_dir, 'MNIST', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # ... (Copiez le reste du code robuste que nous avons fait ensemble) ...
    # Note pour Asmae: Copiez ici le contenu COMPLET de votre fichier src/preprocess.py actuel
    # car il est long et contient la logique de téléchargement automatique.