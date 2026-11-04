"""``python -m evalkit.voicetest.analysis <recording dir> [--vad VERSION]``: writes ``scorecard.json``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .scorecard import write_scorecard

_log = logging.getLogger("evalkit.voicetest")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalkit.voicetest.analysis", description=__doc__)
    parser.add_argument("recording_dir")
    parser.add_argument("--vad", default=None, help="VAD version (default: the one call.json names)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        path = write_scorecard(args.recording_dir, args.vad)
    except (OSError, ValueError) as exc:
        _log.error("%s", exc)
        return 1
    _log.info("%s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
