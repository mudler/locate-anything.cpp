#!/usr/bin/env python3
"""Generate include/la_gguf_keys.h from scripts/gguf_keys.py so the C++ loader
and the Python converter never drift on KV key strings."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import scripts.gguf_keys as K

ROOT = Path(__file__).resolve().parent.parent
HEADER_PATH = ROOT / "include" / "la_gguf_keys.h"

def cident(s):
    return "LA_KV_" + s.replace(".", "_").upper()

def render():
    """Return the full header text. Pure (no I/O) so a test can compare it
    against the committed file to detect drift."""
    idents = [cident(s) for s in K.KV]
    assert len(set(idents)) == len(idents), \
        "cident collision: two K.KV short keys map to the same C identifier"
    lines = ["// AUTO-GENERATED from scripts/gguf_keys.py - do not edit.",
             "#pragma once", ""]
    for short, full in K.KV.items():
        lines.append(f'#define {cident(short)} "{full}"')
    lines.append(f'#define LA_ARCH "{K.ARCH}"')
    return "\n".join(lines) + "\n"

def main():
    HEADER_PATH.write_text(render())
    print("wrote include/la_gguf_keys.h")
if __name__ == "__main__":
    main()
