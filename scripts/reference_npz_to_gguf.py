#!/usr/bin/env python3
"""Convert dumps/reference.npz -> dumps/reference.gguf so the C++ parity harness
(which reads GGUF) can load the M0 gold tensors by name."""
import sys, os
from pathlib import Path
import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent

def main():
    npz = ROOT / "dumps" / "reference.npz"
    out = ROOT / "dumps" / "reference.gguf"
    z = np.load(npz)
    w = gguf.GGUFWriter(str(out), "la_reference")
    n = 0
    for name in z.files:
        a = z[name]
        if name == "input_ids":        # prompt ids — carry as int32 for M3
            w.add_tensor(name, np.ascontiguousarray(a, dtype=np.int32))
            n += 1
            continue
        if a.dtype == np.int64:        # token_stream — skip (not needed in C++)
            continue
        w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
        n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: {n} tensors")

if __name__ == "__main__":
    main()
