import torch
import numpy as np
from torch.nn.functional import softmax
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

# Simplified Top_model to avoid GPT/Module dependencies
class Top_model():
    def __init__(self, model, tokenizer, device = 'cpu', prompt_model=None):
        self.device = device
        self.model = model
        self.prompt_model = prompt_model
        self.tokenizer = tokenizer
        self.word2id = tokenizer.get_vocab()
        self.vocab = [item[0] for item in sorted(self.word2id.items(), key=lambda v:v[1])]

    def encode(self, words):
        return self.tokenizer.encode(words, add_special_tokens=False)

    def get_probs(self, ids):
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs

    def get_probs_generation(self, content_all, additional_bs, additional_bs_mask, content_prev_sep):
        content_all = content_all[:,-500:]
        content_all_mask = torch.ones(content_all.shape).int().to(self.device)
        content_all = content_all.to(self.device)
        content_all2, content_all_mask2 = self.prompt_model.tokenize(content_all, content_all_mask, additional_bs[:content_all.shape[0]], additional_bs_mask[:content_all.shape[0]], content_prev_sep[:content_all.shape[0]], use_fake=False, mode='test')

        with torch.no_grad():
            output = self.model(inputs_embeds=content_all2, attention_mask = content_all_mask2)

        probs = softmax(output.logits, dim = 2).detach().cpu().numpy()
        return probs

    def get_context_array(self, contexts):
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

def generate_beam(model, tokenizer, beam_size: int = 5, embed=None, entry_length=32, temperature=1., stop_token: str = '.', bad_words_ids = None, context_ids = []):
    model.eval()
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)

    with torch.no_grad():
        generated = embed
        for i in range(entry_length):
            outputs = model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                scores_sum = scores[:, None] + logits
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths

            next_token_embed = model.get_input_embeddings()(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    order = scores.argsort(descending=True)
    output_list = np.array([output_list[i] for i in order])
    return torch.tensor(output_list)[0]
