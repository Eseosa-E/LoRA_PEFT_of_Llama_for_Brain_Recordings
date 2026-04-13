import torch
import torch.nn as nn
import random

try:
    from top_model_utils import generate_beam
    from sub_models import Encoding_model
except Exception as e:
    from src.top_model_utils import generate_beam
    from src.sub_models import Encoding_model

class Prompt_model(nn.Module):
    def __init__(self, args, model, tokenizer, device, new_tokens):
        super(Prompt_model, self).__init__()
        self.model = model
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.mse_loss = nn.MSELoss()

        tmp_weights = []
        for new_token in new_tokens:
            new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
            tmp_weight = self.model.get_input_embeddings().weight[new_token_id]
            tmp_weights.append(tmp_weight)

        tmp_weights = torch.stack(tmp_weights)
        self.token_weights = nn.Parameter(tmp_weights.clone().detach(), requires_grad=True)
        self.init_encoding_model()

    def init_encoding_model(self):
        if 'word_embed_size' not in self.args:
             self.args['word_embed_size'] = 4096

        self.encoding_model = Encoding_model(self.args, device=self.device)
        self.encoding_model.to(self.device)

        # PATCH: Keep encoding model in float32 to prevent NaN overflow
        # DO NOT call self.encoding_model.half() — causes NaN loss
        # if 'llama' in self.args.get('model_name', '').lower():
        #     self.encoding_model.half()

    def words2embedding(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def get_prev(self, additional_bs, content_prev_sep):
        if self.args.get('model_name') in ['llama-7b', 'llama-2']:
            return [content_prev_sep[:,:1,:], content_prev_sep[:,1:2,:], additional_bs, content_prev_sep[:,2:,:],]
        else:
            return [content_prev_sep[:,:1,:], additional_bs, content_prev_sep[:,1:,:],]

    def get_tokens(self, content_prev_sep):
        content_prev_sep = self.words2embedding(content_prev_sep)
        # PATCH: use clone() to avoid in-place operation on leaf variable
        if self.token_weights.shape[0] >= 2:
            content_prev_sep = content_prev_sep.clone()
            content_prev_sep[:,-1] = self.token_weights[-1].to(content_prev_sep.dtype)
            content_prev_sep[:,-2] = self.token_weights[-2].to(content_prev_sep.dtype)
        return content_prev_sep

    def tokenize(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=True, mode='train'):
        content_all = self.words2embedding(content_all)
        content_prev_sep = self.get_tokens(content_prev_sep)

        # PATCH: cast to float32 to prevent dtype mismatch with encoding model
        if random.random() > self.args.get('fake_input', 0.0) or not use_fake:
            additional_bs_tokenized = self.encoding_model(additional_bs.float(), position_index=self.args.get('pos', False))
        else:
            additional_bs_tokenized = self.words2embedding(content_all)

        if self.args.get('input_method') == 'without_brain':
            content_all_list = [self.get_prev(additional_bs_tokenized, content_prev_sep)[0]] + [content_all,]
            content_all_mask = torch.cat([additional_bs_mask[:,:1] if additional_bs_mask is not None else content_all_mask[:,:1], content_all_mask], dim=-1)
        else:
            content_all_list = self.get_prev(additional_bs_tokenized, content_prev_sep) + [content_all,]
            brain_mask_len = sum(t.shape[1] for t in content_all_list[:-1])
            brain_mask = torch.ones((content_all.shape[0], brain_mask_len), device=self.device)
            content_all_mask = torch.cat([brain_mask, content_all_mask], dim=-1)

        content_all = torch.cat(content_all_list, dim=-2)
        return content_all, content_all_mask

    def forward(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=True, mode='train'):
        content_all, content_all_mask = self.tokenize(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake, mode)
        output = self.model(inputs_embeds=content_all, attention_mask=content_all_mask)
        return output, content_all_mask

    def pad2left(self, content_prev, content_prev_mask):
        padding_counts = (content_prev_mask == 1).sum(dim=1)
        front_padded_input_embeds = torch.zeros_like(content_prev)
        front_padded_mask = torch.zeros_like(content_prev_mask)
        for i in range(content_prev.size(0)):
            shift = padding_counts[i].item()
            front_padded_input_embeds[i, content_prev.size(1) - shift:] = content_prev[i, :shift]
            front_padded_mask[i, content_prev.size(1) - shift:] = content_prev_mask[i, :shift]
        return front_padded_input_embeds, front_padded_mask

    def generate(self, content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test'):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)

        seq2seqLMoutput = self.model.generate(
            inputs_embeds=content_prev,
            attention_mask=content_prev_mask,
            min_new_tokens=4,
            max_new_tokens=32,
            return_dict_in_generate=True,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        all_truncated_predictions = []
        for i in range(len(seq2seqLMoutput['sequences'])):
            predictions = seq2seqLMoutput['sequences'][i]
            truncated_prediction = [] if predictions[0] == self.tokenizer.eos_token_id else [predictions[0]]
            for t in predictions[1:]:
                if t != self.tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            all_truncated_predictions.append(truncated_prediction)
        return all_truncated_predictions
