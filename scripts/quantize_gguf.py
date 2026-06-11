#!/usr/bin/env python3
"""Quantize a locate-anything gguf (LM matmul weights only).

Walks the source gguf and rewrites it with a new dtype for the heavy LM
matmul tensors (attention projections + FFN + lm.output). EVERYTHING else
is preserved as-is so the rest of the pipeline keeps working.

CRITICAL — these tensors MUST stay f32 (the C++ reads their raw ->data as
f32 on the host, so quantizing them breaks parity):

  lm.tok_embd.weight   (host row-gather in embed_tokens_host)
  vit.pos_emb.weight   (host bicubic in build_patch_pos)

We also keep all vit.*, proj.*, norms (*.norm.weight / *_norm.weight),
and all *.bias as f32. Only weights consumed via ggml_mul_mat are
quantized — ggml_mul_mat dequantizes those on the fly.

Allowlist (regex):

  lm.blk.<N>.attn_(q|k|v|o).weight
  lm.blk.<N>.ffn_(gate|up|down).weight
  lm.output.weight

NOTE: gguf-py does NOT reliably write K-quants (q6_k/q5_k/q4_k). This
script only handles the gguf-py-reliable set: f16, q8_0, q4_0, q5_0.
For q6_k/q5_k/q4_k use the C++ CLI instead:

  ./build/examples/cli/locate-anything-cli quantize <in.gguf> <out.gguf> q4_k

Usage:

  python scripts/quantize_gguf.py <in.gguf> <out.gguf> <type>
"""

import re
import sys
from pathlib import Path

import gguf
import numpy as np

# gguf-py-reliable types only. K-quants (q6_k/q5_k/q4_k) -> use the C++ CLI.
QUANT_MAP = {
    "f16":  gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
}

KQUANT_HINT = {"q6_k", "q5_k", "q4_k"}

# LM matmul weights consumed by ggml_mul_mat. NOTHING else.
QUANTIZABLE = [
    re.compile(r"^lm\.blk\.\d+\.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down)\.weight$"),
    re.compile(r"^lm\.output\.weight$"),
]


def should_quantize(name: str) -> bool:
    return any(p.match(name) for p in QUANTIZABLE)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        print("error: expected 3 args: <in.gguf> <out.gguf> <type>", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    qtype = sys.argv[3].lower()

    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    if qtype in KQUANT_HINT:
        print(f"error: '{qtype}' is a K-quant; gguf-py does not reliably write K-quants.\n"
              f"       Use the C++ CLI instead:\n"
              f"         ./build/examples/cli/locate-anything-cli quantize {src} {out} {qtype}",
              file=sys.stderr)
        return 2

    if qtype not in QUANT_MAP:
        print(f"error: unknown type '{qtype}' (this script: {', '.join(sorted(QUANT_MAP))}; "
              f"K-quants {', '.join(sorted(KQUANT_HINT))} via the C++ CLI)", file=sys.stderr)
        return 2

    target_qt = QUANT_MAP[qtype]

    reader = gguf.GGUFReader(str(src))
    arch = "locate-anything"
    for f in reader.fields.values():
        if f.name == "general.architecture":
            try:
                arch = f.contents()
            except Exception:
                pass
            break

    writer = gguf.GGUFWriter(str(out), arch=arch)

    # ---- copy KV metadata ---------------------------------------------------
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue  # written by GGUFWriter
        try:
            value = f.contents()
        except Exception:
            continue
        if value is None:
            continue
        ft = f.types[0] if f.types else None
        if ft is None:
            continue
        gv = gguf.GGUFValueType
        try:
            if   ft == gv.STRING:  writer.add_string(f.name, value)
            elif ft == gv.UINT8:   writer.add_uint8(f.name, int(value))
            elif ft == gv.INT8:    writer.add_int8(f.name, int(value))
            elif ft == gv.UINT16:  writer.add_uint16(f.name, int(value))
            elif ft == gv.INT16:   writer.add_int16(f.name, int(value))
            elif ft == gv.UINT32:  writer.add_uint32(f.name, int(value))
            elif ft == gv.INT32:   writer.add_int32(f.name, int(value))
            elif ft == gv.UINT64:  writer.add_uint64(f.name, int(value))
            elif ft == gv.INT64:   writer.add_int64(f.name, int(value))
            elif ft == gv.FLOAT32: writer.add_float32(f.name, float(value))
            elif ft == gv.FLOAT64: writer.add_float64(f.name, float(value))
            elif ft == gv.BOOL:    writer.add_bool(f.name, bool(value))
            elif ft == gv.ARRAY:   writer.add_array(f.name, list(value))
        except Exception as e:
            print(f"warning: could not copy KV {f.name} ({ft}): {e}", file=sys.stderr)

    # ---- rewrite tensors ----------------------------------------------------
    n_quantized = 0
    bytes_in = 0
    bytes_out = 0
    for t in reader.tensors:
        src_dtype = t.tensor_type
        np_shape = list(reversed([int(d) for d in t.shape]))
        if should_quantize(t.name):
            arr = gguf.quants.dequantize(t.data, src_dtype).astype(np.float32)
            arr = arr.reshape(np_shape)
            qbytes = gguf.quants.quantize(arr, target_qt)
            writer.add_tensor(t.name, qbytes, raw_dtype=target_qt)
            n_quantized += 1
            bytes_in += t.data.nbytes
            bytes_out += qbytes.nbytes
        else:
            writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=src_dtype)
            bytes_in += t.data.nbytes
            bytes_out += t.data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    pct = 100 * (1 - bytes_out / bytes_in) if bytes_in else 0.0
    print(f"wrote {out}: quantized {n_quantized} LM tensors -> {qtype.upper()}, "
          f"saved {pct:.1f}% of tensor bytes "
          f"(tok_embd/pos_emb/vit/proj/norms/biases kept f32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
