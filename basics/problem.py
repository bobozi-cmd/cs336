
def unicode1():
    assert chr(0) == '\x00'
    
    class A:
        def __str__(self):
            return '__str__'
        def __repr__(self) -> str:
            return '__repr__'
    
    class B:
        def __repr__(self) -> str:
            return '__repr__'

    class C:
        pass

    a = A()
    print(f"A: {a=}, {repr(a)=}")
    
    b = B()
    print(f"B: {b=}, {repr(b)=}")

    c = C()
    print(f"C: {c=}, {repr(c)=}")

    chr(0) # in REPL outputs '\x00'
    print(chr(0)) # null char ''
    "this is a test" + chr(0) + "string" # in REPL outputs 'this is a test\x00string'
    print("this is a test" + chr(0) + "string") # outputs 'this is a teststring'

def unicode2():
    test_string = 'hello! こんにちは! 你好！👋'
    print(test_string.encode('utf-8'))
    print(test_string.encode('utf-16'))
    print(test_string.encode('utf-32'))
    test_ascii = 'abcd ABCD'
    print(test_ascii.encode('utf-8'))
    print(test_ascii.encode('utf-16'))
    print(test_ascii.encode('utf-32'))
    assert 'a'.encode('utf-8') == b'a'
    assert 'a'.encode('utf-16') == b'\xff\xfea\x00'
    assert 'a'.encode('utf-32') == b'\xff\xfe\x00\x00a\x00\x00\x00'

    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode('utf-8') for b in bytestring])
    
    print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
    try:
        print(decode_utf8_bytes_to_str_wrong("你好".encode("utf-8")))
    except Exception as e:
        print(e)

    try:
        print( b'\xc0\x80'.decode('utf-8'))
    except Exception as e:
        print(e)

unicode1()
unicode2()
