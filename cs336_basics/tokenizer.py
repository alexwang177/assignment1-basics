import regex as re
from collections import defaultdict

def train_bpe_tokenizer(input_path, vocab_size, special_tokens):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given input text file.
    
    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired size of the vocabulary.
        special_tokens (list): List of special tokens to include in the vocabulary.
    
    Returns:
        tuple: A tuple containing the vocabulary and merges.
    """

    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)

    vocab = {bytes([i]): i for i in range(256)}
    merges = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Open the file in read mode ('r') and read the contents into a string
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
        chunks = re.split(pattern, content)
        pretoken_freq = defaultdict(int)

        for chunk in chunks:
            # Tokenize the chunk using the regex pattern

            for match in re.finditer(PAT, chunk):
                pretoken = match.group(0)
                pretoken_bytes = list(pretoken.encode('utf-8'))
                pretoken_freq[tuple(pretoken_bytes)] += 1

        while len(vocab) < vocab_size:

            # Find the most frequent byte pair and merge them
            byte_pair_freq = defaultdict(int)
            max_count = 0

            for pretoken_bytes in pretoken_freq.keys():

                for i in range(len(pretoken_bytes) - 1):

                    # print(type(pretoken_bytes))

                    try:
                        b1 = pretoken_bytes[i]
                        b2 = pretoken_bytes[i + 1]
                    except Exception as e:
                        print(pretoken_bytes)
                        raise e

                    byte_pair_freq[(b1, b2)] += pretoken_freq[pretoken_bytes]
                    max_count = max(max_count, byte_pair_freq[(b1, b2)])
            
            max_pairs = [pair for pair, count in byte_pair_freq.items() if count == max_count]
            pair_to_merge = max(max_pairs)
            # print(pair_to_merge)
            
            merges.append(pair_to_merge)

            # Add the merged pair to the vocabulary
            new_idx = len(vocab)
            merged_pair = pair_to_merge[0] + pair_to_merge[1]
            vocab[merged_pair] = new_idx

            print(f"Adding merge: {pair_to_merge} with index {new_idx}")

            # Update the pretoken frequencies with the new merged token
            new_pretoken_freq = defaultdict(int)

            for pretoken_bytes, count in pretoken_freq.items():

                i = 0
                new_bytes = []
                merge_found = False
                while i < len(pretoken_bytes):
                    # Check if current and next byte match the pair to merge

                    # print(type(pretoken_bytes[i]))
                    # print(f"Checking bytes: {pretoken_bytes[i:i+2]} against pair {pair_to_merge}")

                    if (i < len(pretoken_bytes) - 1 and
                        pretoken_bytes[i] == pair_to_merge[0] and
                        pretoken_bytes[i+1] == pair_to_merge[1]):
                        # Replace the pair with the merged token
                        new_bytes.append(merged_pair)
                        i += 2
                        merge_found = True
                    else:
                        new_bytes.append(pretoken_bytes[i])
                        i += 1

                # if merge_found:
                #     print(f"Merged {pretoken_bytes} into {tuple(new_bytes)}")

                new_pretoken_freq[tuple(new_bytes)] += count

            # Update the pretoken frequencies for the next iteration
            pretoken_freq = new_pretoken_freq

    return vocab, merges
    

train_bpe_tokenizer(input_path="../data/TinyStoriesV2-GPT4-valid.txt", vocab_size=300, special_tokens=["<|endoftext|>"])

class BPETokenizer:
    """
    A simple Byte Pair Encoding (BPE) tokenizer.
    This class is used to tokenize text into subword units based on BPE.
    """

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

    def encode(self, text):
        """
        Encode the input text into subword tokens using the BPE vocabulary and merges.
        """
        # Implementation of encoding logic goes here
        pass

    def decode(self, tokens):
        """
        Decode the subword tokens back into the original text.
        """
        # Implementation of decoding logic goes here
        pass