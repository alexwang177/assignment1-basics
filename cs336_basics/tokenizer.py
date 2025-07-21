import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        new_idx = len(vocab)
        vocab[new_idx] = special_token.encode('utf-8')

    merges = []

    # Open the file in read mode ('r') and read the contents into a string
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
        chunks = re.split(pattern, content)
        pretoken_freq = defaultdict(int)

        for chunk in chunks:
            # Tokenize the chunk using the regex pattern

            for match in re.finditer(PAT, chunk):
                pretoken = match.group(0)
                pretoken_bytes = pretoken.encode('utf-8')
                pretoken_bytes_tuple = tuple(bytes([b]) for b in pretoken_bytes)
                pretoken_freq[pretoken_bytes_tuple] += 1

        while len(vocab) < vocab_size:

            # Find the most frequent byte pair and merge them
            byte_pair_freq = defaultdict(int)
            max_count = 0

            for pretoken_bytes in pretoken_freq.keys():

                for i in range(len(pretoken_bytes) - 1):

                    try:
                        b1 = pretoken_bytes[i]
                        b2 = pretoken_bytes[i + 1]
                        # print(f"Checking bytes: {b1}, {b2} ---- types: {type(b1)}, {type(b2)}")
                    except Exception as e:
                        print(pretoken_bytes)
                        raise e

                    byte_pair_freq[(b1, b2)] += pretoken_freq[pretoken_bytes]
                    max_count = max(max_count, byte_pair_freq[(b1, b2)])
            
            max_pairs = [pair for pair, count in byte_pair_freq.items() if count == max_count]
            pair_to_merge = max(max_pairs)
            
            merges.append(pair_to_merge)

            # Add the merged pair to the vocabulary
            new_idx = len(vocab)
            merged_pair = pair_to_merge[0] + pair_to_merge[1]
            vocab[new_idx] = merged_pair

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


class BPETokenizer:
    """
    A simple Byte Pair Encoding (BPE) tokenizer.
    This class is used to tokenize text into subword units based on BPE.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens.sort(key=len, reverse=True)  # Sort special tokens by length for regex matching
        self.token_to_id = {token: idx for idx, token in vocab.items()}

    def encode(self, text):
        """
        Encode the input text into subword tokens using the BPE vocabulary and merges.
        """
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = "(" + "|".join(escaped_tokens) + ")"

            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        final_tokens = []

        for chunk in chunks:
            # Tokenize the chunk using the regex pattern
            # print("chunk: ", chunk)
            if chunk in self.special_tokens:
                # If the chunk is a special token, directly append its ID
                final_tokens.append(self.token_to_id[chunk.encode('utf-8')])
                continue

            for match in re.finditer(PAT, chunk):
                pretoken = match.group(0)
                pretoken_bytes = pretoken.encode('utf-8')
                pretoken_bytes_tuple = tuple(bytes([b]) for b in pretoken_bytes)

                # print("\nbefore: ", pretoken_bytes_tuple)

                while True:

                    merged_applied = False
                    for merge in self.merges:
                        
                        for i in range(len(pretoken_bytes_tuple) - 1):
                            b1 = pretoken_bytes_tuple[i]
                            b2 = pretoken_bytes_tuple[i + 1]

                            if (b1, b2) == merge:
                                # print(f"Merging {b1} and {b2}")
                                # Merge the bytes
                                pretoken_bytes_tuple = (
                                    *pretoken_bytes_tuple[:i],
                                    b1 + b2,
                                    *pretoken_bytes_tuple[i + 2:]
                                )
                                merged_applied = True
                                break

                    if not merged_applied:
                        break

                # print("after: ", pretoken_bytes_tuple)

                # At this point, pretoken_bytes_tuple should be fully merged
                for b in pretoken_bytes_tuple:
                    if b in self.token_to_id:
                        final_tokens.append(self.token_to_id[b])
                    else:
                        # If the byte is not in the vocabulary, we can handle it as needed
                        # For simplicity, we will skip it here
                        continue
    
        return final_tokens

    def encode_iterable(self, iterable):
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens):
        """
        Decode the subword tokens back into the original text.
        """
        decoded_tokens = []

        for token_id in tokens:
            if token_id in self.vocab:
                decoded_tokens.append(self.vocab[token_id])
            else:
                # Handle unknown tokens
                decoded_tokens.append(b"<unk>")

        return b"".join(decoded_tokens).decode('utf-8', errors='replace')

# tokenizer = BPETokenizer(
#     vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
#     merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')], 
#     # special_tokens=["<|endoftext|>"]
# )
# token_ids = tokenizer.encode('the cat ate')
# print(token_ids)
# print(tokenizer.decode(token_ids))