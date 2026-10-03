"""bench/false_done.py: telling format failures from logic failures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from false_done import failure_kind  # noqa: E402

CHECK = "assert slugify('Héllo') == 'hello'"


def test_missing_requested_function_is_format():
    assert failure_kind("Traceback ...\nNameError: name 'slugify' is not defined", CHECK) == "format"
    out = "The worker's code did not compile:\nSyntaxError\nNameError: name 'slugify' is not defined"
    assert failure_kind(out, CHECK) == "format"


def test_wrong_value_is_logic_even_if_loading_crashed():
    # granite4.2:8b and qwen3.5:2b on the A5000 were mislabelled format by the old rule
    out = ("The worker's code raised an error while loading (definitions before the error are still used):\n"
           "EOFError\nTraceback ...\nFailed: slugify('Héllo') returned 'helloworld', expected 'hello'")
    assert failure_kind(out, CHECK) == "logic"


def test_undefined_helper_is_logic():
    assert failure_kind("Traceback ...\nNameError: name 'helper' is not defined", CHECK) == "logic"
