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
    max_generated_tokens: Optional[int] = None,
    temp: Optional[int] = None, 
    top_p: Optional[int] = None,
    device: str = "mps"
) -> List[str]:
    
    tokenizer = get_tokenizer()
    
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
        # logging.info(f"Logits: {logits}"

        probs = softmax(x=logits, i=-1)
        next_token_id = torch.argmax(probs)
        # logging.info(f"Next token probs: {probs}")
        # logging.info(f"Next token shape: {next_token_id.shape}")
        # logging.info(f"Next token ID: {next_token_id}")

        output_tokens.append(next_token_id.item())
        input_tokens = torch.cat((input_tokens, torch.unsqueeze(next_token_id, 0).to(device)), 0)

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
        top_p=args.top_p
    )

    print(output)

if __name__ == "__main__":
    main()
