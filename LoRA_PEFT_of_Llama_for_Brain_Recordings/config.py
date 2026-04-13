import os
def get_config():
    args = {
        # --- CRITICAL IDENTIFIERS ---
        'task_name': 'Huth_example',
        'model_name': 'llama-7b',
        'llm_model_path': 'meta-llama/Llama-2-7b-hf',
        # --- PATHS ---
        'dataset_path': '/workspace/dataset',
        'checkpoint_path': '/workspace/results/checkpoint/',
        # --- TRAINING HYPERPARAMETERS ---
        'batch_size': 2,
        'num_epochs': 10,
        'lr': 1e-4,
        'dropout': 0.1,
        'random_number': 0,
        # --- ARCHITECTURE ---
        'brain_model': 'mlp',
        'brain_embed_size': 1000,
        'word_embed_size': 4096,
        'pos': False,
        'fake_input': 0.0,
        'input_method': 'with_brain',
        'mode': 'train',
        'use_noise': False,
        'load_check_point': False,
        'num_layers': 2,
        'activation': 'relu',
        'normalized': False,
    }
    args['test_trail_ids'] = [0, 0]
    args['valid_trail_ids'] = [0, 0]
    if not os.path.exists(args['checkpoint_path']):
        os.makedirs(args['checkpoint_path'], exist_ok=True)
    return args
