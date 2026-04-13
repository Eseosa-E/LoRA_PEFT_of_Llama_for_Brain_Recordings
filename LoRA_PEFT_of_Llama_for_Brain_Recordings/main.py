import os
import pickle
import random
import numpy as np
import torch
import importlib
from config import get_config
from data import FMRI_dataset
import model
from model import Decoding_model
from huggingface_hub import login

# 1. AUTHENTICATION
login(token='#*generate own token*')

# 2. SETUP REPRODUCIBILITY
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 3. CONFIGURATION
args = get_config()

# --- CRITICAL OVERRIDES ---
args['llm_model_path'] = 'meta-llama/Llama-2-7b-hf'
args['dataset_path'] = '/workspace/dataset'
args['task_name'] = 'Huth_example'
args['mode'] = 'train'
args['fake_input'] = 0.0
args['input_method'] = 'with_brain'
args['use_noise'] = False
args['cuda'] = '0'
args['pos'] = False
args['data_size'] = 100
args['brain_embed_size'] = 256

huth_pickle_file = os.path.join(args['dataset_path'], 'example.pca1000.wq.pkl.dic')

# 4. LOAD DATASET FILE
if not os.path.exists(huth_pickle_file):
    raise FileNotFoundError(f"File not found at: {huth_pickle_file}")

with open(huth_pickle_file, 'rb') as f:
    input_dataset = pickle.load(f)

# 5. INITIALIZE MODEL & DATASET
importlib.reload(model)
decoding_model = Decoding_model(args)
dataset = FMRI_dataset(
    input_dataset,
    args,
    tokenizer=decoding_model.tokenizer,
    decoding_model=decoding_model
)
print(f">>> Initialization successful. Dataset and JSON mapped.")

# 6. PERFORM TRAINING
print(">>> Starting LoRA Fine-tuning...")
decoding_model.train(dataset.train_dataset, dataset.valid_dataset)
print("--- Training Complete ---")
