import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from settings import model_name2path, model2hidden
    from model_utils import Prompt_model
except:
    from src.settings import model_name2path, model2hidden
    from src.model_utils import Prompt_model

class Decoding_model:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model_path = model_name2path.get(args['model_name'], args['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(">>> Materializing model weights...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GiB"}
        )

        base_model = prepare_model_for_kbit_training(base_model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(base_model, peft_config)

        self.new_tokens = ["<brain/>", "</brain>"]
        self.tokenizer.add_tokens(self.new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        args['word_embed_size'] = model2hidden.get(args['model_name'], 4096)
        self.prompt_model = Prompt_model(args, self.model, self.tokenizer, self.device, self.new_tokens)

    def calculate_metrics(self, preds, targets):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1
        m = {'r1': [], 'r2': [], 'rL': [], 'bleu': []}

        for p, t in zip(preds, targets):
            if not p.strip(): continue
            s = scorer.score(t, p)
            m['r1'].append(s['rouge1'].fmeasure)
            m['r2'].append(s['rouge2'].fmeasure)
            m['rL'].append(s['rougeL'].fmeasure)
            m['bleu'].append(sentence_bleu([t.split()], p.split(), smoothing_function=smoothing))
        return {k: np.mean(v) for k, v in m.items()}

    def train(self, tr_dataset, vl_dataset):
        print("🚀 Training started on RTX 3090...")
        batch_size = self.args.get('batch_size', 1)
        train_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(vl_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.prompt_model.parameters(), lr=float(self.args.get('lr', 1e-4)))

        self.prompt_model.train()
        for epoch in range(int(self.args.get('num_epochs', 10))):

            # --- Training ---
            total_loss = 0
            self.prompt_model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                batch_gpu = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
                c_prev, add_bs, c_prev_sep, c_true, c_prev_mask, c_true_mask, c_all, c_all_mask, d_id = batch_gpu

                output, content_all_mask = self.prompt_model(
                    c_all, c_all_mask, add_bs, None, c_prev_sep, use_fake=False, mode='train'
                )

                logits = output.logits
                shift_logits = logits[..., :-1, :].contiguous()

                # PATCH: Label alignment — pad labels to match extended
                # sequence length caused by brain tokens being prepended
                seq_len = shift_logits.shape[1]
                label_len = c_all.shape[1] - 1
                if seq_len > label_len:
                    pad = torch.full((c_all.shape[0], seq_len - label_len),
                                     self.tokenizer.pad_token_id,
                                     dtype=c_all.dtype, device=c_all.device)
                    shift_labels = torch.cat([pad, c_all[..., 1:]], dim=1).contiguous()
                else:
                    shift_labels = c_all[..., 1:seq_len+1].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                loss.backward()

                # PATCH: Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

            # --- Validation ---
            if len(vl_dataset) > 0:
                self.prompt_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in valid_dataloader:
                        batch_gpu = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
                        c_prev, add_bs, c_prev_sep, c_true, c_prev_mask, c_true_mask, c_all, c_all_mask, d_id = batch_gpu

                        output, content_all_mask = self.prompt_model(
                            c_all, c_all_mask, add_bs, None, c_prev_sep, use_fake=False, mode='train'
                        )

                        logits = output.logits
                        shift_logits = logits[..., :-1, :].contiguous()

                        # PATCH: Same label alignment fix for validation loop
                        seq_len = shift_logits.shape[1]
                        label_len = c_all.shape[1] - 1
                        if seq_len > label_len:
                            pad = torch.full((c_all.shape[0], seq_len - label_len),
                                             self.tokenizer.pad_token_id,
                                             dtype=c_all.dtype, device=c_all.device)
                            shift_labels = torch.cat([pad, c_all[..., 1:]], dim=1).contiguous()
                        else:
                            shift_labels = c_all[..., 1:seq_len+1].contiguous()

                        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                        val_loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

                print(f"Epoch {epoch} | Val Loss: {val_loss / len(valid_dataloader):.4f}")

            # PATCH: Save LoRA weights only (not full model) to save disk space
            checkpoint_path = os.path.join(self.args['checkpoint_path'], f'epoch_{epoch}.pt')
            lora_state = {k: v for k, v in self.prompt_model.state_dict().items()
                         if 'lora' in k.lower() or 'encoding_model' in k.lower()
                         or 'token_weights' in k.lower()}
            torch.save(lora_state, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
