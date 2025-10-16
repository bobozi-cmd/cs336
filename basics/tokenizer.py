import argparse
from collections import defaultdict
import mmap
from pathlib import Path
import random
import time
from typing import BinaryIO
import regex as re
import warnings
import json
import os
import multiprocessing
import heapq
from tqdm import tqdm

debug_mode = os.environ.get('DEBUG', False)


class GPT2Converter:
    def __init__(self):
        self.gpt2_unicode_map = self._gpt2_bytes_to_unicode()
        self.gpt2_unicode_rmap = {v : k for k, v in self.gpt2_unicode_map.items()}

    def _gpt2_bytes_to_unicode(self) -> dict[int, str]:
        # 收集可直接表示的字符
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
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


class BPEHeap():
    def __init__(self, token_group: list[list[bytes]], vocab: dict[int, bytes]) -> None:
        self.token_group = token_group
        self.vocab = vocab # read-only, update by tokenizer
        self.pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        # 记录pair的位置: 假设token_group = [[100, 101, 102, 103], [99, 100, 101, 202]]
        # (100, 101) = [(0, 1), (1, 2)] 表示 (100, 101) 出现在第0个group的第1个位置和第1个group的第2个位置
        # self.pair_positions: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self.heap: list[PairItem] = []

        for i, token_indices in enumerate(token_group):
            for pos in range(len(token_indices) - 1):
                pair = (token_indices[pos], token_indices[pos+1])
                self.pair_counts[pair] += 1
                # self.pair_positions[pair].append((i, pos))
        
        for k, v in self.pair_counts.items():
            heapq.heappush(self.heap, PairItem(k, (self.vocab[k[0]], self.vocab[k[1]]), v))

    
    def pop(self) -> tuple[int, int]:
        if debug_mode:
            print('-'*100)
        while self.heap:
            top = heapq.heappop(self.heap)
            if top.pair not in self.pair_counts:
                continue
            if self.pair_counts[top.pair] != top.count:
                if debug_mode:
                    print(f"Failed: {top} -> {self.pair_counts[top.pair]}")
                if self.pair_counts[top.pair] > 1:
                    heapq.heappush(self.heap, PairItem(top.pair, top.pair_bytes, self.pair_counts[top.pair]))
                continue
        
            if debug_mode:
                print(f"Found {top}")
            return top.pair
        return None

    def merge_and_update(self, top, new_index):
        new_count: dict[tuple[int, int], int] = defaultdict(int)
        for group_idx, indices in enumerate(self.token_group):
            i = 0
            if top[0] in indices and top[1] in indices:
                new_indices = []
                while i < len(indices):
                    if i + 1 < len(indices) and indices[i] == top[0] and indices[i+1] == top[1]:
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
                self.token_group[group_idx] = new_indices
        
        for k, v in new_count.items():
            if debug_mode:
                print(f"Add {PairItem(k, (self.vocab[k[0]], self.vocab[k[1]]), v)}")
            self.pair_counts[k] = v
            heapq.heappush(self.heap, PairItem(k, (self.vocab[k[0]], self.vocab[k[1]]), v))


    # def merge_and_update(self, top, new_index):
    #     position_by_group = defaultdict(list)

    #     for group_idx, pos in self.pair_positions[top]:
    #         position_by_group[group_idx].append(pos)
        
    #     new_count: dict[tuple[int, int], int] = defaultdict(int)

    #     for group_idx, positions in position_by_group.items():
    #         # inplace modify token_group
    #         indices: list[bytes] = self.token_group[group_idx]
    #         # 将positions倒序合并, 不然前面做了合并会导致后面的idx被打乱
    #         positions.sort(reverse=True)
    #         last_merged_pos = -2

    #         for i in positions:
    #             # 检查当前pos是否有效, 可能因为前面的merge, 导致当前pos失效
    #             if i >= len(indices) - 1 or i <= last_merged_pos:
    #                 continue 
    #             if indices[i] != top[0] or indices[i+1] != top[1]:
    #                 continue

    #             # # 更新左侧pair
    #             if i > 0:
    #                 new_count[(indices[i-1], new_index)] += 1
    #                 self.pair_counts[(indices[i-1], indices[i])] -= 1
    #                 self.pair_positions[(indices[i-1], new_index)].append((group_idx, i-1))

    #             if i + 2 < len(indices):
    #                 new_count[(new_index, indices[i+2])] += 1
    #                 self.pair_counts[(indices[i+1], indices[i+2])] -= 1
    #                 self.pair_positions[(new_index, indices[i+2])].append((group_idx, i))

    #             # 合并
    #             indices[i] = new_index
    #             del indices[i+1]
    #             last_merged_pos = i
            
    #         # 更新group, 同时刷新当前group的position
    #         # self.token_group[group_idx] = indices
    #         # for pos in range(len(indices) - 1):
    #         #     self.pair_positions[(indices[pos], indices[pos+1])].append((group_idx, pos))

    #     # print(f"For Merge {top}: {self.token_group[:1000:50]} ...")

    #     # 清理已经合并的
    #     if top in self.pair_counts:
    #         del self.pair_counts[top]
    #     if top in self.pair_positions:
    #         del self.pair_positions[top]

    #     for k, v in new_count.items():
    #         if debug_mode:
    #             print(f"Add {PairItem(k, (self.vocab[k[0]], self.vocab[k[1]]), v)}")
    #         self.pair_counts[k] = v
    #         heapq.heappush(self.heap, PairItem(k, (self.vocab[k[0]], self.vocab[k[1]]), v))


class BPETokenizer():
    def __init__(self, input_file: Path, vocab_size: int, special_tokens: list[str]):
        self.input_file = input_file
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        base_vocab_size = len(special_tokens) + 256
        assert vocab_size >= base_vocab_size, f"vocab size must be larger than {base_vocab_size}"

    def train(self, n_proc: int = 2, n_sample: int = 200):

        # 初始化词汇表
        self.vocab: dict[int, bytes] = {len(self.special_tokens) + i: bytes([i]) for i in range(256)}
        for i, sp_token in enumerate(self.special_tokens):
            self.vocab[i] = sp_token.encode('utf-8')

        self.rvocab: dict[int, bytes] = { v:k for k, v in self.vocab.items() }

        # 加载 并 采样数据, down scale
        # 在足够大的样本中，高频字符对的相对频率分布是稳定的
        # 一旦样本达到一定规模，继续增加数据对最终词汇表的影响很小
        text = self._load_and_sample_data(n_sample, '<|endoftext|>')

        # 分割文档
        pattern = "|".join([re.escape(token) for token in self.special_tokens])
        documents: list[str] = [part for part in re.split(pattern, text) if part]

        # 并行预分词
        token_group = self._parallel_pretokenize(documents, n_proc)
        print(f"After PreTokenize, got {len(token_group):,} tokens group")

        # 初始化索引结构
        bpe_heap = BPEHeap(token_group, self.vocab)
        self.merges: list[tuple[bytes, bytes]] = []
        n_merges = self.vocab_size - len(self.vocab)

        # 开始训练
        for i in tqdm(range(n_merges), desc="Traning BPE"):
            top = bpe_heap.pop()
            if top is None:
                break
            
            index1, index2 = top
            if debug_mode:
                print(f"Found {top} : {self.vocab[index1]}, {self.vocab[index2]}")

            # 创建新的token
            new_index = len(self.vocab)
            new_token_bytes = self.vocab[index1] + self.vocab[index2]
            self.merges.append((self.vocab[index1], self.vocab[index2]))
            self.vocab[new_index] = new_token_bytes
            self.rvocab[new_token_bytes] = new_index
            
            # 合并
            bpe_heap.merge_and_update(top, new_index)
        return self.vocab, self.merges

    def _load_and_sample_data(self, n_sample: int, delim: str) -> str:
        try:
            with open(self.input_file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    documents = []
                    start = 0
                    while start < len(mm):
                        end = mm.find(delim.encode('utf-8'), start)
                        if end == -1: # last doc
                            doc = mm[start:].decode('utf-8', errors='replace')
                        else:
                            doc = mm[start:end].decode('utf-8', errors='replace')
                        
                        if doc:
                            documents.append(doc)
                        
                        start = end + len(delim) if end != -1 else len(mm)

                    if len(documents) > n_sample:
                        print(f"Down scale from {len(documents)} to {n_sample}")
                        documents = random.sample(documents, n_sample)
                    
                return delim.join(documents)
        except Exception as e:
            raise IOError(f"Load dataset Failed: {e}")

    def _parallel_pretokenize(self, documents: list[str], n_proc: int) -> list[list[int]]:
        bytes_group = [] # doc -> list[bytes]
        if n_proc <= 1:
            bytes_group = [pre_tokenize(doc) for doc in documents]
        else:
            with multiprocessing.Pool(n_proc) as pool:
                bytes_group = list(tqdm(
                    pool.imap(pre_tokenize, documents, chunksize=50),
                    total=len(documents),
                    desc="PreTokenize",
                    mininterval=1
                ))

        token_group = []
        for doc_token_bytes in bytes_group:
            for token_byte in doc_token_bytes:
                token_group.append([self.rvocab[bytes([b])] for b in token_byte])

        return token_group


def save(vocab, merges, special_tokens):
    converter = GPT2Converter()
    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    with open("test_vocab.json", "w") as f:
        reference_vocab = {(converter.from_unicode(bs) if bs not in special_tokens_bytes else special_tokens[idx]) : idx for idx, bs in vocab.items()}
        json.dump(reference_vocab, f, ensure_ascii=False)

    with open("test_merges.txt", "w") as f:
        for (b1, b2) in merges:
            if debug_mode:
                print(list(b1), list(b2), converter.from_unicode(b1), converter.from_unicode(b2), file=f)
            else:
                print(converter.from_unicode(b1), converter.from_unicode(b2), file=f)


def evaluation(special_tokens):
    vocab_size = 10000
    n_proc = 8
    sample_size = 22000

    train_path = Path('/home/zwb/Jobs/cs336/data/TinyStoriesV2-GPT4-train.txt')
    valid_path = Path('/home/zwb/Jobs/cs336/data/TinyStoriesV2-GPT4-valid.txt')

    assert train_path.exists() and valid_path.exists()

    print("🚀 开始训练")
    start_time = time.time()
    train_tokenizer = BPETokenizer(train_path, vocab_size, special_tokens)
    train_vocab, train_merges = train_tokenizer.train(n_proc, sample_size)
    print(f"\n✅ 训练完成! 耗时: {time.time() - start_time:.2f}秒")

    # 小规模验证 (使用验证集的10%)
    print("\n🔬 小规模验证")
    valid_tokenizer = BPETokenizer(train_path, vocab_size, special_tokens)
    valid_vocab, valid_merges = valid_tokenizer.train(n_proc, 2)

    # 分析结果
    print("\n📊 训练结果")
    print(f"训练词汇表大小: {len(train_vocab):,}")
    print(f"训练合并操作数: {len(train_merges):,}")
    print(f"验证词汇表大小: {len(valid_vocab):,}")
    print(f"验证合并操作数: {len(valid_merges):,}")

    # 比较词汇表重叠率
    train_tokens = set(train_vocab.values())
    valid_tokens = set(valid_vocab.values())
    overlap = train_tokens & valid_tokens
    print(f"\n📈 词汇表重叠率: {len(overlap)/len(train_tokens):.1%}")

    def evaluate_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], test_text: str):
        """简单评估分词器效果"""
        print("\n🔍 分词器评估")
        sample_text = test_text[:200] + "..." if len(test_text) > 200 else test_text
        print(f"样例文本: {sample_text}")
        
        # 简单统计
        unique_tokens = set(vocab.values())
        print(f"词汇表大小: {len(vocab):,}")
        print(f"唯一token数: {len(unique_tokens):,}")
        print(f"合并操作数: {len(merges):,}")

    # 加载验证集样例进行评估
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_text = f.read(1000)  # 读取前1000字符用于评估
    evaluate_tokenizer(train_vocab, train_merges, valid_text)

    return train_vocab, train_merges


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=Path, required=True)
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--eval", default=False, action="store_true")
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    if args.eval:
        vocab, merges = evaluation(special_tokens)
    else:
        vocab_size = 500
        n_proc = 8
        sample_size = 22000

        tokenizer = BPETokenizer(args.file, vocab_size, special_tokens)
        vocab, merges = tokenizer.train(n_proc, sample_size)

    if args.save:
        save(vocab, merges, special_tokens)

    import psutil
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"💾 峰值内存使用: {mem_usage:.2f} GB")

