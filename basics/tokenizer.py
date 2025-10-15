import argparse
from collections import defaultdict
from pathlib import Path
import regex as re
import warnings
import json
import os

debug_mode = os.environ.get('DEBUG', False)


class GPT2Converter:
    def __init__(self):
        self.gpt2_unicode_map = self._gpt2_bytes_to_unicode()
        self.gpt2_unicode_rmap = {v : k for k, v in self.gpt2_unicode_map.items()}

    def _gpt2_bytes_to_unicode(self) -> dict[int, str]:
        """
        Returns a mapping between every possible byte (an integer from 0 to 255) to a
        printable unicode string character representation. This function is taken
        from the GPT-2 code.

        For example, `chr(0)` is `\x00`, which is an unprintable character:

        >>> chr(0)
        '\x00'
        >>> print(chr(0))

        As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
        The bytes that are visually printable keep their original string representation [1].
        For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
        Note in particular that the space character `chr(32)` becomes `d[32]`, which
        returns 'Ġ'.

        For unprintable characters, the function shifts takes the integer representing
        the Unicode code point of that character (returned by the Python `ord`) function
        and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
        ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
        string representation of the space.

        This function can simplify the BPE implementation and makes it slightly easier to
        manually inspect the generated merges after they're serialized to a file.
        """
        # These 188 integers can used as-is, since they are not whitespace or control characters.
        # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
        # 收集可直接表示的字符
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        # now get the representations of the other 68 integers that do need shifting
        # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
        # Get printable representations of the remaining integers 68 integers.
        n = 0
        for b in range(2**8):
            if b not in bs:
                # If this integer isn't in our list of visually-representable
                # charcters, then map it to the next nice character (offset by 256)
                # 处理需要转换的字符，将其偏移256个位置，使其可以被Unicode所打印
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        d = dict(zip(bs, characters)) # 原始字节 -> 对应的Unicode字符
        return d

    def to_unicode(self, inp: str) -> bytes:
        return bytes([self.gpt2_unicode_rmap[c] for c in inp])
    
    def from_unicode(self, inp: bytes) -> str:
        return ''.join([self.gpt2_unicode_map[bi] for bi in inp])


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize(text) -> list[bytes]:
    # 按照PAT的规则，将文本切分城多个list[bytes]
    str_tokens = re.findall(PAT, text)
    bytes_tokens = [tok.encode('utf-8') for tok in str_tokens]
    return bytes_tokens


def detokenize(vocab: dict[int, bytes], tokens: list[int]):
    ret = b''
    for idx in tokens:
        ret += vocab[idx]
    return ret


def merge(token_group: list[list[int]], pair: tuple[int, int], new_index: int) -> list[list[int]]:
    new_group = []
    for indices in token_group:
        new_indices = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        new_group.append(new_indices)
    return new_group


class BPETokenizer():
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        
        self.merges: list[tuple[bytes, bytes]] = []
        self.stoi: dict[bytes, int] = {}
        self.itos: dict[int, bytes] = {}
        self.merges_rank: dict[tuple[bytes, bytes], int] = {}

        # 初始化vocab, 先加载 special tokens, 然后再加载 0-255
        for i, token_byte in enumerate(self.special_tokens_bytes):
            self.stoi[token_byte] = i
            self.itos[i] = token_byte
        
        offset = len(self.special_tokens_bytes)
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])

        # self.vocab = self.itos.copy() # 用来序列化
        # self.merges_rank = {} # fast lookup
        # # (p1, p2) -> new_token_id
        # self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    def train_slow(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as fp:
            text = fp.read()

        blocks = [] # 每个 special token 分割出一个block，多个block组成一个chunk
        if len(self.special_tokens) > 0:
            pattern = "|".join([re.escape(token) for token in self.special_tokens])
            blocks = re.split(pattern, text)
        else:
            blocks = [text]

        # {b'<|endoftext|>': 0, b'\x00': 1, ... }
        inital_vocab_rmap = {v : k for k, v in self.itos.items()}
        # 对每个block进行pre-tokenize, 产生出的每个token, 存下其bytes list
        token_group: list[list[int]] = []
        for block in blocks:
            if block in self.special_tokens or not block:
                print(block)
                continue

            for token_byte in pre_tokenize(block):
                token_group.append([inital_vocab_rmap[bytes([b])] for b in token_byte])
            
        # print(f"PreTokenize: {token_group[:5]} ...")

        n_merges = self.vocab_size - len(self.itos)
        for i in range(n_merges):
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for token_indices in token_group:
                for index1, index2 in zip(token_indices[:], token_indices[1:]):
                    counts[(index1, index2)] += 1
            
            if len(counts) == 0:
                warnings.warn("Found empty counts")
                break

            pair = max(counts, key=lambda k: (counts.get(k), self.itos[k[0]], self.itos[k[1]]))
            index1, index2 = pair
            
            if debug_mode:
                sorted_count = sorted(counts.items(), key=lambda p: p[1], reverse=True)
                print(sorted_count[:5])
                print(pair, ":", detokenize(self.itos, [index1, index2]))

            # 新增 pair, 更新group
            new_index = len(self.itos)
            p1_bytes, p2_bytes = self.itos[index1], self.itos[index2]
            new_token_bytes = p1_bytes + p2_bytes

            self.merges.append((p1_bytes, p2_bytes))
            self.stoi[new_token_bytes] = new_index
            self.itos[new_index] = new_token_bytes

            token_group = merge(token_group, pair, new_index)
            # print(f"PreTokenize: {token_group[:5]} ...")
        
        return self.itos, self.merges



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=Path, required=True)
    args = parser.parse_args()

    assert pre_tokenize("some text that i'll pre-tokenize") == [b'some', b' text', b' that', b' i', b"'ll", b' pre', b'-', b'tokenize']
    
    t1 = BPETokenizer(500, ['<|endoftext|>'])
    vocab, merges = t1.train_slow(args.file)

    rvocab = {v : k for k, v in vocab.items()}
    converter = GPT2Converter()

    # for idx, bs in vocab.items():
    #     if bs not in t1.special_tokens_bytes:
    #         print(idx, repr(bs), converter.from_unicode(bs))
    #     else:
    #         print(idx, repr(bs), t1.special_tokens[idx])


    with open("test_vocab.json", "w") as f:
        reference_vocab = {(converter.from_unicode(bs) if bs not in t1.special_tokens_bytes else t1.special_tokens[idx]) : idx for idx, bs in vocab.items()}
        json.dump(reference_vocab, f, ensure_ascii=False)

    with open("test_merges.txt", "w") as f:
        for (b1, b2) in merges:
            if debug_mode:
                print(list(b1), list(b2), converter.from_unicode(b1), converter.from_unicode(b2), file=f)
            else:
                print(converter.from_unicode(b1), converter.from_unicode(b2), file=f)

