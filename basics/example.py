
def unicode_encodings():
    test_string = 'hello! こんにちは!'
    utf8_encoded = test_string.encode('utf-8')
    assert utf8_encoded == b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
    assert type(utf8_encoded) == bytes
    assert list(utf8_encoded) == [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
    assert len(test_string) == 13
    assert len(utf8_encoded) == 23
    assert utf8_encoded.decode('utf-8') == 'hello! こんにちは!'

unicode_encodings()


