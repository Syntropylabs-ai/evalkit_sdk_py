"""``python -m evalkit.voicetest ...``: the same commands as ``evalkit voice-test ...``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(prog="python -m evalkit.voicetest"))
