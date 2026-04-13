#Dataset
import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json
import copy
import os

# Using a fallback scaler if sklearn is not available in the environment
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class MyStandardScaler:
        def __init__(self):
            self.mean = 0
            self.std = 0
        def fit(self, X):
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        def transform(self, X):
            return (X - self.mean) / (self.std + 1e-8)
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
    StandardScaler = MyStandardScaler

class Splited_FMRI_dataset(Dataset):
    """Standard PyTorch Dataset for fMRI samples"""
    def __init__(self, inputs, most_epoch=-1, args=None):
        self.device = torch.device(f"cuda:{args['cuda']}")
        self.inputs = inputs
        self.most_epoch = most_epoch
        self.args = args

    def __len__(self):
        if self.most_epoch > -1:
            return min(self.most_epoch, len(self.inputs))
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        return (
            item['content_prev'],       # Past text context tokens
            item['additional_bs'],     # Neural (fMRI) signal
            item['content_prev_sep'],   # Optional separators
            item['content_true'],      # Target text tokens
            item['content_prev_mask'],
            item['content_true_mask'],
            item['content_all'],       # Combined tokens for training
            item['content_all_mask'],
            item['id']
        )

class FMRI_dataset():
    """Logic for loading Huth dataset and prepping LLaMA-compatible inputs"""

    def pack_info(self, content_prev, additional_bs, content_true, trail_id, id):
        # LLaMA specific: We ensure space prefix management and max length constraints
        content_all = self.tokenizer(
            content_prev.strip() + ' ' + content_true,
            max_length=self.args['prev_mask_len'] + self.args['max_generate_len'],
            truncation=True, return_tensors='pt', add_special_tokens=False, padding='max_length'
        )

        # Target content encoding
        content_true_encoded = self.tokenizer(
            content_true,
            max_length=self.args['max_generate_len'],
            add_special_tokens=False, truncation=True, return_tensors='pt', padding='max_length'
        )

        # Context content encoding
        content_prev_encoded = self.tokenizer(
            content_prev.strip(),
            max_length=self.args['prev_mask_len'],
            truncation=True, return_tensors='pt', add_special_tokens=False, padding='max_length'
        )

        return {
            'content_prev': content_prev_encoded['input_ids'][0],
            'content_prev_mask': content_prev_encoded['attention_mask'][0],
            'additional_bs': torch.tensor(additional_bs, dtype=torch.float32),
            # Special tags used in some brain-to-text architectures
            'content_prev_sep': self.tokenizer(['<brain/>', '</brain>'], return_tensors='pt')['input_ids'][0],
            'content_true': content_true_encoded['input_ids'][0],
            'content_true_mask': content_true_encoded['attention_mask'][0],
            'trail_id': trail_id,
            'content_all': content_all['input_ids'][0],
            'content_all_mask': content_all['attention_mask'][0],
            'id': id
        }

    def __init__(self, input_dataset, args, tokenizer, decoding_model=None):
        self.decoding_model = decoding_model
        self.args = args
        self.inputs = []
        self.tokenizer = tokenizer

        if args['normalized']:
            self.scaler = StandardScaler()

        id2info = {}
        tmp_id = 0

        # Restricted to Huth logic only
        if 'huth' in args['task_name'].lower(): # Changed this line to make it case-insensitive
            subject_name = args['task_name'].split('_')[1]

            # TODO: update this path - Suggestion: Use /content/drive/MyDrive/dataset/Huth.json
           # if not os.path.exists(huth_info_path):
           #     print(f"Warning: Huth info file not found at {huth_info_path}")
           #     data_info2 = list(input_dataset.keys())
           # else:
           #     data_info2 = json.load(open(huth_info_path))
# Point this to where your Huth.json actually is on Drive
            huth_info_path = os.path.join(args['dataset_path'], 'Huth.json')
            if not os.path.exists(huth_info_path):
                print(f"Warning: Huth info file not found at {huth_info_path}. Falling back to keys.")
                data_info2 = list(input_dataset.keys())
            else:
                data_info2 = json.load(open(huth_info_path))
            # TODO: update this path - Suggestion: Ensure your .pkl.dic files are in this directory
            huth_data_file = f"{args['dataset_path']}/{subject_name}.pca1000.wq.pkl.dic"
            ds_dataset = pickle.load(open(huth_data_file, 'rb'))

            # Remap ds_dataset from numeric keys to story name keys
            num_stories = len(data_info2)
            remapped_ds = {}
            for idx in range(len(ds_dataset)):
                story_name = data_info2[idx % num_stories]
                if story_name not in remapped_ds:
                    remapped_ds[story_name] = {'fmri': []}
                for row in ds_dataset[idx]['fmri']:
                    remapped_ds[story_name]['fmri'].append(row)
            import numpy as np
            for story in remapped_ds:
                remapped_ds[story]['fmri'] = np.array(remapped_ds[story]['fmri'])
            ds_dataset = remapped_ds

            content_true2idx = {}
            for sid, story in enumerate(input_dataset.keys()):
                if story not in data_info2:
                    continue

                for item_id, item in enumerate(input_dataset[story]):
                    # Iterate through word sequences to build context pairs
                    for k in range(1, len(item['word'])):
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0, k)])
                        # Map fMRI indices to actual voxel data
                        additional_bs = np.array([ds_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']

                        if len(content_true.strip()) == 0:
                            continue

                        # Data splitting logic
                        if content_true not in content_true2idx:
                            content_true2idx[content_true] = random.random()

                        trail_id = content_true2idx[content_true]

                        id2info[tmp_id] = {'story': story, 'item_id': item_id, 'k': k}

                        # Package for the LLaMA/LoRA pipeline
                        packed_info = self.pack_info(
                            content_prev.lower(),
                            additional_bs,
                            content_true.lower(),
                            trail_id,
                            id=tmp_id
                        )
                        tmp_id += 1

                        # Filter out empty samples
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)
        else:
            raise ValueError("Only Huth dataset is supported in this configuration.")

        # Finalize split (Train/Valid/Test)
        self.pack_data_from_input(args)

        # Save mapping for reproducibility/debugging
        # TODO: update this path - Ensure checkpoint_path is set in config
        json.dump(id2info, open(os.path.join(self.args['checkpoint_path'], 'id2info.json'), 'w'))

    def pack_data_from_input(self, args):
        self.train, self.test, self.valid = [], [], []
        test_ids = args['test_trail_ids']
        valid_ids = args['valid_trail_ids']

        for item in self.inputs:
            if test_ids[0] < item['trail_id'] <= test_ids[1]:
                self.test.append(item)
            elif valid_ids[0] < item['trail_id'] <= valid_ids[1]:
                self.valid.append(item)
            else:
                self.train.append(item)

        # Sample training data if restricted size is requested
        if args['data_size'] != -1:
            random.shuffle(self.train)
            self.train = self.train[:args['data_size']]

        # Create PyTorch Dataset objects
        self.train_dataset = Splited_FMRI_dataset(self.train, args=args)
        self.valid_dataset = Splited_FMRI_dataset(self.valid, args=args) if len(self.valid) > 0 else Splited_FMRI_dataset(self.test, args=args)
        self.test_dataset = Splited_FMRI_dataset(self.test, args=args)
