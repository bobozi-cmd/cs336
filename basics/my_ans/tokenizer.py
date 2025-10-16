import argparse
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO
import regex as re
import warnings
import json
import os
import multiprocessing
import heapq

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



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 1. 均分当前文件成多个chunk
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 2. 找到当前chunk下面最近的一个special token，重新设置边界
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 3. 可能出现两个chunk下面最近的一个special token是同一个，所以要去重
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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


class PairItem():
    def __init__(self, pair: tuple[int, int], pair_bytes: tuple[bytes, bytes], count: int):
        self.pair = pair
        self.pair_bytes = pair_bytes
        self.count = count

    def __gt__(self, other: "PairItem"): # reversed for heapq
        if self.count == other.count:
            return self.pair_bytes < other.pair_bytes
        return self.count < other.count
    
    def __eq__(self, other: "PairItem"):
        return self.pair == other.pair and self.count == self.count
    
    def __repr__(self):
        return f"Pair<{self.pair}, {self.pair_bytes}, {self.count}>"


class BPETokenizerTrainer():
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

    @staticmethod
    def _pretokenize_chunk(args):
        chunk_text, special_tokens, itos = args
        # 1. find_chunk_boundaries() -> list[chunk_text]
        # 2. pretokenize_chunk() -> list[list[int]]
        # 3. reduce -> list[list[int]]
        blocks = [] # 每个 special token 分割出一个block，多个block组成一个chunk
        if len(special_tokens) > 0:
            pattern = "|".join([re.escape(token) for token in special_tokens])
            blocks = re.split(pattern, chunk_text)
        else:
            blocks = [chunk_text]

        # {b'<|endoftext|>': 0, b'\x00': 1, ... }
        inital_vocab_rmap = {v : k for k, v in itos.items()}
        # 对每个block进行pre-tokenize, 产生出的每个token, 存下其bytes list
        token_group: list[list[int]] = []
        for block in blocks:
            if block in special_tokens or not block:
                continue

            for token_byte in pre_tokenize(block):
                token_group.append([inital_vocab_rmap[bytes([b])] for b in token_byte])
        return token_group

    def _collect_freqs(self, token_group: list[list[int]]):
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for token_indices in token_group:
            for index1, index2 in zip(token_indices[:], token_indices[1:]):
                counts[(index1, index2)] += 1
        return counts

    def train_slow(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as fp:
            text = fp.read()

        token_group: list[list[int]] = []
        token_group.extend(self._pretokenize_chunk((text, self.special_tokens, self.itos)))
        # print(f"PreTokenize: {token_group[:5]} ...")
        return {}, []

        n_merges = self.vocab_size - len(self.itos)
        for i in range(n_merges):
            counts: dict[tuple[int, int], int] = self._collect_freqs(token_group)
            
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
    
    def _merge_and_update(self, heap: list[PairItem], token_group: list[list[int]], pair: tuple[int, int], new_index: int) -> list[list[int]]:
        new_count: dict[tuple[int, int], int] = defaultdict(int)
    
        new_group = []
        for indices in token_group:
            i = 0
            if pair[0] in indices and pair[1] in indices:
                new_indices = []
                while i < len(indices):
                    if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
                        new_indices.append(new_index)
                        if i > 0:
                            new_count[(indices[i-1], new_index)] += 1
                            self.pair_counts[(indices[i-1], indices[i])] -= 1
                        if i + 2 < len(indices):
                            new_count[(new_index, indices[i+2])] += 1
                            self.pair_counts[(indices[i+1], indices[i+2])] -= 1
                        i += 2
                    else:
                        new_indices.append(indices[i])
                        i += 1
                new_group.append(new_indices)
            else:
                new_group.append(indices)

        for k, v in new_count.items():
            if debug_mode:
                print(f"Add {PairItem(k, (self.itos[k[0]], self.itos[k[1]]), v)}")
            self.pair_counts[k] = v
            heapq.heappush(heap, PairItem(k, (self.itos[k[0]], self.itos[k[1]]), v))

        return new_group

    def train_fast(self, file_path: Path, n_chunks: int = 4):
        # with open(file_path, "r", encoding="utf-8") as fp:
        #     text = fp.read()
        token_group: list[list[int]] = []

        chunks = []
        with open(file_path, 'rb') as f:
            num_processes = max(1, n_chunks)
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
        
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            result = pool.map(BPETokenizerTrainer._pretokenize_chunk, iterable=[(chunk, self.special_tokens, self.itos) for chunk in chunks])

        for res in result:
            token_group.extend(res)
        return {}, []

        n_merges = self.vocab_size - len(self.itos)
        # 大根堆，维护频次最高的pair
        # 可能的情况: (s, t) = 10, ('a', 's') = 10, pop + merge (s, t) 会导致 ('a', 's') 的频次失效(如, ['a', 's', 't'] -> ['a', 'st'])
        # 为了解决这个问题, 需要额外维护一个pair_count, 每次merge的时候更新 (原始的pair--)
        # 在取出top的时候, 和 pair_count 里面的值进行对比, 一致就使用, 不一致就更新之后重新插入heaq, 再次取top
        heap: list[PairItem] = []
        
        self.pair_counts: dict[tuple[int, int], int] = self._collect_freqs(token_group)
        for k, v in self.pair_counts.items():
            heapq.heappush(heap, PairItem(k, (self.itos[k[0]], self.itos[k[1]]), v))
        
        # for i in range(n_merges):
        while len(self.itos) < self.vocab_size:
            pair = heapq.heappop(heap)
            if self.pair_counts[pair.pair] != pair.count:
                if debug_mode:
                    print(f"Failed: {pair} -> {self.pair_counts[pair.pair]}")
                heapq.heappush(heap, PairItem(pair.pair, pair.pair_bytes, self.pair_counts[pair.pair]))
                continue
            
            if debug_mode:
                print(f"Found {pair}")
            index1, index2 = pair.pair

            new_index = len(self.itos)
            p1_bytes, p2_bytes = self.itos[index1], self.itos[index2]
            new_token_bytes = p1_bytes + p2_bytes

            self.merges.append((p1_bytes, p2_bytes))
            self.stoi[new_token_bytes] = new_index
            self.itos[new_index] = new_token_bytes

            token_group = self._merge_and_update(heap, token_group, pair.pair, new_index)
            
        return self.itos, self.merges

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=Path, required=True)
    args = parser.parse_args()

    # assert pre_tokenize("some text that i'll pre-tokenize") == [b'some', b' text', b' that', b' i', b"'ll", b' pre', b'-', b'tokenize']
    
    t1 = BPETokenizerTrainer(500, ['<|endoftext|>'])
    # vocab, merges = t1.train_slow(args.file)
    vocab, merges = t1.train_fast(args.file)

    # lis = [PairItem((0, 0), (b'\x00', b'\x00'), 10), PairItem((120, 0), (b'\x00\x01', b'\x00'), 9), PairItem((12, 0), (b'a\x01', b'\x00'), 10)]
    # lis.sort()
    # print(lis)
    # heapq.heapify(lis)
    # while len(lis) > 0:
    #     top = heapq.heappop(lis)
    #     print(top)


    converter = GPT2Converter()

    # for idx, bs in vocab.items():
    #     if bs not in t1.special_tokens_bytes:
    #         print(idx, repr(bs), converter.from_unicode(bs))
    #     else:
    #         print(idx, repr(bs), t1.special_tokens[idx])


    # with open("test_vocab.json", "w") as f:
    #     reference_vocab = {(converter.from_unicode(bs) if bs not in t1.special_tokens_bytes else t1.special_tokens[idx]) : idx for idx, bs in vocab.items()}
    #     json.dump(reference_vocab, f, ensure_ascii=False)

    # with open("test_merges.txt", "w") as f:
    #     for (b1, b2) in merges:
    #         if debug_mode:
    #             print(list(b1), list(b2), converter.from_unicode(b1), converter.from_unicode(b2), file=f)
    #         else:
    #             print(converter.from_unicode(b1), converter.from_unicode(b2), file=f)

