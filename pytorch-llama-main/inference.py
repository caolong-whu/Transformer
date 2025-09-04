from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs

class LLaMA:
    
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint file found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            print(f'Loaded checkpoint in {time.time() - prev_time:.2f}s')
            prev_time = time.time()
            
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
            
        model = Transformer(model_args).to(device)
        
        if load_model:
            # # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time()-prev_time:.2f}s')
            
        return LLaMA(model, tokenizer, model_args)
    
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # Convert each prompt into tokens, add BOS and not EOS
        # (B, seq_len)
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequenc length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        # True if the token reaches EOS!
        eos_reached_flags = torch.tensor([False] * batch_size, device=device) # (B,)
        # True if the token is not equal to pad id
        prompt_tokens_mask = tokens != pad_id # (B, total_len)
        
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            # Start predict
            with torch.no_grad():
                # (B, 1, vocab_size)
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            # Softmax:
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1) # (B, vocab_size)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy stragedy
                next_token = torch.argmax(logits[:,-1], dim=-1)
            next_token = next_token.reshape(-1) # (B)
            # Only replace the token if it is a pad token
            # Why cur_pos starts from 1? -> KV Cache
            next_token = torch.where(prompt_tokens_mask, tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # (B)
            eos_reached_flags = eos_reached_flags | (~prompt_tokens_mask[:, cur_pos] & next_token == self.tokenizer.eos_id())
            if all(eos_reached_flags):
                break
        
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
            
        return (out_tokens, out_text)
            
            
        
    
    def _sample_top_p(self, probs: torch.Tensor, p: int):
        # (B, vocab_size) -> (B, vocab_size)
        probs_sort, probs_index = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        # (1, 2, 3, 4) -> (1, 3, 6, 10)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size) True if the cumulative sum > p (not include current position)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token index from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1) # (B, 1)
        # Get the token position in the vocablualry corresponding to the sampled index
        next_token = torch.gather(probs_index, -1, next_token)
        return next_token
        
    
if __name__ == '__main__':
    # Set the seed for random generator
    torch.manual_seed(0)
    
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
        
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]    
        
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )
    
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=60))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-' * 50)