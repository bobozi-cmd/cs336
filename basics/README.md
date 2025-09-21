# Assignment 1 (Basics)
- 课程目标：构建训练一个标准Transformer LM的所有组建, 包括：
    - BPE(Byte-pair encoding) tokenizer
    - Transformer language model
    - The cross-entropy loss function & AdamW optimizer
    - The training loop, with support for serializing and loading model and optimizer state

- 实现限制：不允许使用 `torch.nn, torch.nn.functional, torch.optim`, 除了：
    - `torch.nn.Parameter`
    - `torch.nn` 中的容器类 (如 `Module, ModuleList, Sequential, ...`)
    - `torch.optim.Optimizer` 基类

## BPE Tokenizer
- 将任意的Unicode字符串表示为一串bytes来训练tokenizer
- Unicode 标准将字符映射成code points, 其vocabulary非常大(约150K)且稀疏 (character-level), 训练起来效率低; 先讲字符转成用utf-8编码的字节序列 (byte-level), 可以将vocabulary压缩到 256 起步, 后面再根据BPE的规则拼接出常用的字节对, 省空间, 大大降低了稀疏的问题.
- word-level 的 tokenizer 会遇到 out-of-vocabulary 的情况 (遇到训练时没遇过的 token), 而 character-level/byte-level 会导致输入的文本被编码成很长的序列, 会带来更多的计算和数据的更长期的依赖. 一种中间选择是 subword tokernizer, 用更大的 vocabulary size 换取更好的压缩, 比如 b'the' 出现的频率很高, 就为 b'the' 分配新的 vocab-id 来单独代表这个token. BPE encoding 就是一种选择 subword 的算法.
- BPE Tokenizer 的训练包括三个步骤:
    - **Vocabulary initialization**: 建立256个bytes和vocab-id的映射
    - **Pre-tokenization**: 遍历一遍训练语料来对相邻bytes做合并以及统计频次开销太大, 并且会将语义相近的token切分成无关的token(如 "dog." 和 "dog!"). 可以通过预先粗分一遍词(如 `line.split(" ")` 按 word 粗分), 然后再在这些词内部做合并, 节省算力又避免标点把近义词拆散. 


### Problem
- unicode1
    
    (a) `assert chr(0) == '\x00'`
    
    (b) `print()`调用的是`__str__()`, `repr()`调用的是`__repr__()`, 如果没实现 `__str__()`, `print()`调用`__repr__()`
    
    (c) `chr(0)` 在REPL里面直接输出表现为 `0x00`, 用 `print()` 则是空字符
- unicode2

    (a) utf-8可以将每个 ASCII 字符保留为单个字节, 而 utf-16, utf-32 会将每个 ASCII 字符编码成多个字节.
    ```python
    assert 'a'.encode('utf-8') == b'a'
    assert 'a'.encode('utf-16') == b'\xff\xfea\x00'
    assert 'a'.encode('utf-32') == b'\xff\xfe\x00\x00a\x00\x00\x00'
    ```

    (b) `decode_utf8_bytes_to_str_wrong` 只能 decode 字符的编码长度为1的情况, 大于1就会出错

    (c) 变种的Unicode中使用双字节的 `0xc0 0x80` 表示空字符, 在标准unicode中是非法的


