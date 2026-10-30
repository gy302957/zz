from knowledge_base import get_string_md5, save_md5, check_md5


def test_get_string_md5_deterministic():
    a = get_string_md5("住院报销比例")
    b = get_string_md5("住院报销比例")
    assert a == b
    assert len(a) == 32


def test_get_string_md5_different_inputs():
    a = get_string_md5("住院报销")
    b = get_string_md5("门诊报销")
    assert a != b


def test_save_md5_and_check(tmp_path):
    path = str(tmp_path / "test_md5.txt")
    save_md5("abc123", path)
    assert check_md5("abc123", path) is True
    assert check_md5("def456", path) is False


def test_check_md5_creates_file_if_missing(tmp_path):
    path = str(tmp_path / "new_md5.txt")
    result = check_md5("xyz", path)
    assert result is False
    import os
    assert os.path.exists(path)