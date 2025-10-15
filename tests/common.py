from __future__ import annotations

import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


# 将256个可能的字节值（0-255）映射到可打印的Unicode字符，便于文本处理和可视化
# Unicode字符集 低位多是不可打印的控制字符，偏移256后会变成可打印的拉丁文扩展字符
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
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


def gpt2_str_to_unicode_repr(converter, inp: str) -> bytes:
    rconverter = {v: k for k, v in converter.items()}
    return bytes([rconverter[c] for c in inp])


def unicode_str_to_gpt2_str(converter, inp: bytes) -> bytes:
    return ''.join([converter[bi] for bi in inp])


if __name__ == "__main__":
    converter = gpt2_bytes_to_unicode()
    for idx, c in converter.items():
        if idx != ord(c):
            print(repr(chr(idx)), c)

    print(unicode_str_to_gpt2_str(converter, b'\x00a\x01'))
    print(repr(gpt2_str_to_unicode_repr(converter, 'Āaā')))
    