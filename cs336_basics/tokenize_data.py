from cs336_basics.tokenizer import *

import numpy as np

def main():
    vocab_input_path = "tests/fixtures/gpt2_vocab.json"
    merges_input_path = "tests/fixtures/gpt2_merges.txt"
    special_tokens = ['<|endoftext|>']

    vocab, merges = Tokenizer.from_file(vocab_input_path, merges_input_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        contents = f.read()
        token_ids = tokenizer.encode(contents)
        array = np.array(token_ids, dtype=np.int64)
        np.save("data/tiny_stories_train_dataset.npy", array)
        

if __name__ == "__main__":
    main()
