from collections import defaultdict
import regex as re

def unicode_encodings():
    test_string = 'hello! こんにちは!'
    utf8_encoded = test_string.encode('utf-8')
    assert utf8_encoded == b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
    assert type(utf8_encoded) == bytes
    assert list(utf8_encoded) == [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
    assert len(test_string) == 13
    assert len(utf8_encoded) == 23
    assert utf8_encoded.decode('utf-8') == 'hello! こんにちは!'

def BPE_tokenizer_training():
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    assert re.findall(PAT, "some text that i'll pre-tokenize") == ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
    assert max([('A', 'B'), ('A', 'C'), ('B', "ZZ"), ('BA', 'A')]) == ('BA', 'A')

    corpus = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    """

    print(f"{'>'*30} Vocabulary-Init {'<'*30}")
    vocabulary = [chr(i) for i in range(256)]
    vocabulary.insert(0, '<|endoftext|>')
    print(vocabulary[: 10], "...")

    print(f"{'>'*30} Pre-tokenization {'<'*30}")
    def split_str2bytes(s: str) -> str:
        tokens = [c for c in s]
        return ','.join(tokens)

    freq_tb = defaultdict(lambda : 0)
    for line in corpus.splitlines():
        for word in line.strip().split():
            freq_tb[split_str2bytes(word)] += 1
    print(freq_tb.items())

    
    def merge(freq_tb: dict):
        nonlocal vocabulary
        freq_pair_tb = defaultdict(lambda : 0)
        for k, v in freq_tb.items():
            bytes_ = k.split(",")
            for i in range(len(bytes_) - 1):
                freq_pair_tb[bytes_[i] + bytes_[i + 1]] += v
        print(freq_pair_tb.items())
        first = sorted(freq_pair_tb.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)[0]
        print(f"Merge {first}")
        vocabulary.append(first[0])
        ret = defaultdict(lambda : 0)
        for k, v in freq_tb.items():
            bytes_ = k.split(",")
            new_word = ""
            for i in range(len(bytes_) - 1):
                if bytes_[i] + bytes_[i + 1] == first[0]:
                    if i > 0:
                        new_word += ",".join(bytes_[:i])
                        new_word += ","
                    new_word += first[0]
                    if i + 2 < len(bytes_):
                        new_word += ","
                        new_word += ",".join(bytes_[i+2:])
                    break
            if new_word == "":
                ret[k] = v
            else:
                ret[new_word] = v
            
        return ret

    for i in range(6):
        print(f"{'>'*30} Merge-{i} {'<'*30}")
        freq_tb = merge(freq_tb)
        print("After merge:", freq_tb.items())
    
    print(vocabulary[: 5], "...", vocabulary[len(vocabulary) - 10:])

    def tokenize(word: str) -> list:
        nonlocal vocabulary
        st = 0
        tokens = []
        while st < len(word):
            for et in range(len(word))[::-1]:
                # print(word[st:et+1])
                if word[st:et+1] in vocabulary:
                    tokens.append(word[st:et+1])
                    st = et + 1
                    break

        return tokens

    print(f"least -> {tokenize('least')}")

def bpe_encoding():
    corpus = "the cat ate"
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at'
    }
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

    rvocab = { v:k for k, v in vocab.items()}
    prefix_merges: dict[bytes, list[bytes]] = defaultdict(list)
    for b1, b2 in merges:
        prefix_merges[b1].append(b2)
    
    subwords = ["the", " cat", " ate"]
    tokens_id = []
    for subword in subwords:
        token = [bytes([ei]) for ei in subword.encode('utf-8')]
        while True:
            found = False
            for i in range(len(token) - 1):
                if token[i] in prefix_merges and token[i+1] in prefix_merges[token[i]]:
                    found = True
                    token[i] = vocab[rvocab[token[i] + token[i+1]]]
                    del token[i+1]
                    break
            if not found:
                break
        
        token_id = []
        for tok in token:
            token_id.append(rvocab[tok])

        print([bytes([ei]) for ei in subword.encode('utf-8')], '->', token, '->', token_id)
        tokens_id.extend(token_id)

    bytes_list = list(map(vocab.get, tokens_id))
    string = b"".join(bytes_list).decode('utf-8')
    print(string)

        


if __name__ == "__main__":
    # unicode_encodings()
    # BPE_tokenizer_training()
    bpe_encoding()

