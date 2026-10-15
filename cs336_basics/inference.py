import torch
import argparse
import logging

from typing import List, Optional

from cs336_basics.utils import *
from cs336_basics.tokenizer import *

logging.basicConfig(                                        
    level=logging.INFO,                                     
    format="%(asctime)s | %(levelname)s | %(message)s",     
) 

def get_tokenizer():
    vocab_input_path = "tests/fixtures/gpt2_vocab.json"
    merges_input_path = "tests/fixtures/gpt2_merges.txt"
    special_tokens = ['<|endoftext|>']

    vocab, merges = Tokenizer.from_file(vocab_input_path, merges_input_path)
    return Tokenizer(vocab, merges, special_tokens)


def decode(
    model: torch.nn.Module,
    prompt: str, 
    context_length: int,
    temp: float,
    max_generated_tokens: int,
    top_p: Optional[float] = None,
    device: str = "mps"
) -> str:
    
    tokenizer = get_tokenizer()
    EOS_ID = tokenizer.encode("<|endoftext|>")[0]
    
    logging.info(f"User prompt: [{prompt}]")

    input_tokens = tokenizer.encode(prompt)
    input_tokens  = torch.tensor(input_tokens, dtype=torch.long, device=device)

    logging.info(f"Token IDs: {input_tokens}")

    if len(input_tokens) > context_length:
        input_tokens = input_tokens[-context_length:]

    output_tokens = []

    while len(output_tokens) < max_generated_tokens:

        with torch.inference_mode():
            logits = model(input_tokens)[-1]

        # logging.info(f"Logits Shape: {logits.shape}")
        # logging.info(f"Logits: {logits}")

        probs = softmax(x=logits, i=-1, temp=temp)


        if top_p is not None:
            # Top-p sampling

            assert top_p > 0 and top_p <= 1.0

            logging.info(f"Using top-p sampling...")
            probs_values, probs_indices = torch.sort(probs, descending=True)
            prefix_sum = torch.cumsum(probs_values, dim=-1)

            logging.info(f"Sorted probs: {probs_values}")
            logging.info(f"Prefix sum: {prefix_sum}")

            insert_idx = torch.searchsorted(prefix_sum, top_p)
            logging.info(f"Insert idx: {insert_idx}")

            # Truncate probability distribution of tokens
            probs_values = probs_values[:insert_idx + 1]
            lookup_idx = torch.multinomial(probs_values, 1)
            next_token_id = probs_indices[lookup_idx]

        else:
            # Normal sampling
            next_token_id = torch.multinomial(probs, 1)
            logging.info(f"Next token probs: {probs}")

        logging.info(f"Next token shape: {next_token_id.shape}")
        logging.info(f"Next token ID: {next_token_id}")

        next_token_int = next_token_id.item()

        if next_token_int == EOS_ID:
            break

        output_tokens.append(next_token_int)
        input_tokens = torch.cat((input_tokens, next_token_id), 0)

        input_tokens = input_tokens[-context_length:]

        logging.info(f"Output tokens: {len(output_tokens)} Max tokens: {max_generated_tokens}")
        
    return tokenizer.decode(output_tokens)
    
def main():
    parser = argparse.ArgumentParser()

    # hardware
    parser.add_argument("--device", default="mps")

    # model
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0) 

    # decoding
    parser.add_argument("--max_generated_tokens", type=int, default=16)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.2)
   
    args = parser.parse_args()

    # initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.emb_dim,
        d_ff=args.ffn_dim,
        rope_theta=args.rope_theta
    )
    model = model.to(args.device)

    output = decode(
        model=model,
        prompt="Hi my name is Alex.",
        context_length=args.context_length,
        max_generated_tokens=args.max_generated_tokens,
        temp=args.temp,
        top_p=args.top_p,
        device=args.device
    )

    print(output)

if __name__ == "__main__":
    main()
