"""Run pytest as the CPU CI job would see it: no Triton, no flash-attn, no CUDA."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ARENO_CHECK_TOKEN_IDS"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
for name in ("triton", "flash_attn", "flash_attn_2_cuda"):
    sys.modules[name] = None

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
