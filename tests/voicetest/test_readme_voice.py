"""The README's voice testing section is the PyPI page's: it names no plan, price or minute allowance,
and sends the reader to the contact page (the pricing decision of 2026-11-19)."""

from __future__ import annotations

import os
import re

from .support import SDK_ROOT


def _voice_section() -> str:
    with open(os.path.join(SDK_ROOT, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    start = text.index("## Testing your voice bot")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_the_voice_section_names_no_plan_price_or_allowance():
    section = _voice_section()
    assert not re.search(r"\b(?:Free|Pro|Enterprise)\b", section)
    assert not re.search(r"\$\s?\d|/pricing|paid plan|upgrade", section, re.I)
    assert not re.search(r"\b\d[\d,]* (?:SyntropyLabs model )?minutes (?:a|per|each) (?:month|day)\b", section)
    assert "https://syntropylabs.ai/contact" in section


def test_platform_minutes_are_set_up_by_syntropylabs_not_the_organization():
    section = _voice_section()
    # Only SyntropyLabs grants model minutes; an org's own admin has no setting for them.
    assert "administrator" not in section
    assert "where SyntropyLabs has set up model minutes for your organization" in section
