#!/usr/bin/env python3
r"""image_race — compose an animation FROM FRAMES with Pillow, then encode with
ffmpeg. The counterpart to render-card.sh (HTML->still PNG): when your content is
IMAGES — annotated detections, before/after, two models labeling the same photo —
a terminal recorder can't show them, so you draw each frame yourself and stitch.

This sample is a two-pane "detection race": the same image side by side, boxes
popping in over each engine's REAL measured time, the faster one filling first and
getting a star, both ending on the identical set of boxes + a summary card.

Why this instead of record.sh? record.sh CAPTURES a live terminal/GUI window;
this RENDERS frames directly — no Xvfb, no Docker, frame-precise, and it can put
real images on screen. Same output container (mp4/gif), different source.

The honest-timing rule (same as the duel): the seconds drawn on screen are the
real measured numbers; --dilate only sets PLAYBACK speed, so a 69 s race is
watchable in ~11 s without faking anything.

  python3 image_race.py                          # synthetic sample, 16:9
  python3 image_race.py --layout square          # 1:1 for social
  python3 image_race.py --layout vertical        # 9:16 for reels/stories
  python3 image_race.py --spec my.json --out my.mp4

Spec JSON (all fields optional except engines+boxes; a synthetic scene is drawn
if "image" is missing):
  {
    "image": "scene.png", "img_w": 960, "img_h": 540,   # box coord space
    "note": "identical boxes", "link": "github.com/you/tool",
    "boxes": [["person",[x1,y1,x2,y2]], ["car",[...]]],
    "engines": [
      {"label":"engine A","device":"CPU","proc_s":22.3,"rate":"11.6 tok/s","accent":"teal"},
      {"label":"engine B","device":"CPU","proc_s":69.1,"rate":"3.7 tok/s","accent":"slate"}
    ]
  }
"""
import argparse, json, math, subprocess, tempfile, hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BG, INK, DIM, GREEN, GOLD = (13, 17, 23), (215, 221, 229), (110, 118, 129), (70, 194, 102), (240, 200, 90)
ACCENTS = {"teal": (62, 200, 224), "slate": (148, 163, 178), "amber": (255, 207, 86),
           "green": (126, 231, 135), "violet": (175, 145, 245), "rose": (244, 130, 150)}
LAYOUTS = {"cols": (1280, 720, "h"), "square": (1080, 1080, "v"), "vertical": (1080, 1920, "v")}
FPS, REVEAL_BY = 20, 0.85


def fontp(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
def font(sz, bold=True):
    try: return ImageFont.truetype(fontp(bold), sz)
    except Exception: return ImageFont.load_default()


def label_color(lab):
    h = int(hashlib.md5(lab.encode()).hexdigest(), 16)
    pal = list(ACCENTS.values())
    return pal[h % len(pal)]


def synthetic_scene(w=960, h=540):
    """A self-contained stand-in 'photo' so the recipe runs with zero assets."""
    img = Image.new("RGB", (w, h), (28, 36, 48))
    d = ImageDraw.Draw(img)
    for i in range(h):                       # vertical gradient sky/ground
        d.line([(0, i), (w, i)], fill=(28 + i*30//h, 36 + i*40//h, 48 + i*20//h))
    d.rectangle([0, int(h*0.7), w, h], fill=(40, 46, 40))          # ground
    d.rectangle([int(w*0.62), int(h*0.45), int(w*0.95), int(h*0.72)], fill=(120, 70, 60))  # "car"
    for cx in (int(w*0.18), int(w*0.40)):    # two "people"
        d.ellipse([cx-12, int(h*0.40)-12, cx+12, int(h*0.40)+12], fill=(210, 180, 160))
        d.rectangle([cx-14, int(h*0.40)+12, cx+14, int(h*0.70)], fill=(60, 90, 140))
    return img


def load_spec(path):
    if path:
        spec = json.loads(Path(path).read_text())
    else:
        spec = {"note": "identical boxes", "link": "github.com/mudler/recorder-for-agents",
                "query": "person · car",
                "boxes": [["person", [150, 216, 230, 378]], ["person", [355, 216, 415, 378]],
                          ["car", [595, 243, 912, 389]]],
                "engines": [{"label": "engine-fast", "device": "CPU", "proc_s": 22.3, "rate": "11.6 tok/s", "accent": "teal"},
                            {"label": "engine-base", "device": "CPU", "proc_s": 69.1, "rate": "3.7 tok/s", "accent": "slate"}]}
    if spec.get("image") and Path(spec["image"]).exists():
        img = Image.open(spec["image"]).convert("RGB")
    else:
        img = synthetic_scene()
    spec.setdefault("img_w", img.width); spec.setdefault("img_h", img.height)
    for e in spec["engines"]:
        e["_c"] = ACCENTS.get(e.get("accent", "teal"), ACCENTS["teal"]) if isinstance(e.get("accent"), str) else tuple(e["accent"])
    return spec, img


def fit(img, box_w, box_h):
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width*s)), max(1, int(img.height*s))), Image.LANCZOS), s


def draw_pane(cv, rect, base, scale, spec, eng, frac, t_real, winner):
    d = ImageDraw.Draw(cv); ox, oy, pw, ph = rect; accent = eng["_c"]
    done = frac >= 1.0
    iw, ih = base.size
    ix, iy = ox + (pw - iw)//2, oy + 6
    cv.paste(base, (ix, iy)); d.rectangle([ix-1, iy-1, ix+iw, iy+ih], outline=accent, width=2)
    sx, sy = iw/spec["img_w"], ih/spec["img_h"]
    boxes = spec["boxes"]; n = len(boxes)
    shown = min(n, int(math.floor(frac/REVEAL_BY*n + 1e-6)))
    fl = font(max(12, iw//40), False)
    for i in range(shown):
        lab, (x1, y1, x2, y2) = boxes[i]; c = label_color(lab)
        r = [ix+x1*sx, iy+y1*sy, ix+x2*sx, iy+y2*sy]
        d.rectangle(r, outline=c, width=3)
        tw = d.textlength(lab, font=fl)
        d.rectangle([r[0], r[1]-18, r[0]+tw+8, r[1]], fill=c); d.text((r[0]+4, r[1]-17), lab, fill=BG, font=fl)
    cy = iy + ih + 12; fs = font(20); ft = font(16, False); fl2 = font(15, False)
    d.text((ix, cy), eng["label"], fill=accent, font=fs)
    d.text((ix + d.textlength(eng["label"], font=fs) + 10, cy+2), eng.get("device", ""), fill=DIM, font=ft)
    by = cy + 30; d.rounded_rectangle([ix, by, ix+iw, by+8], 4, fill=(34, 41, 50))
    d.rounded_rectangle([ix, by, ix+int(iw*min(1.0, frac)), by+8], 4, fill=accent)
    sy2 = by + 15
    if done:
        s = f"✓ {eng['proc_s']:.1f} s    {eng.get('rate','')}"; d.text((ix, sy2), s, fill=INK, font=fs)
        if winner: d.text((ix + d.textlength(s, font=fs) + 16, sy2), "★ fastest", fill=GOLD, font=fs)
    else:
        d.text((ix, sy2), f"▸ {min(t_real, eng['proc_s']):.1f} s", fill=accent, font=fs)


def panes(W, H, orient, top=96):
    if orient == "h":
        pw = (W - 80)//2; return [(40, top, pw, H-top-54), (40+pw, top, pw, H-top-54)]
    ph = (H - top - 54)//2; return [(60, top, W-120, ph), (60, top+ph+20, W-120, ph)]


def query_bar(cv, W, y, query, accent):
    """A 'search pill' showing what was asked — e.g. the detection prompt."""
    d = ImageDraw.Draw(cv); fq = font(18); fl = font(16, False)
    lab = "locate"
    lw = d.textlength(lab + "  ", font=fl); qw = d.textlength(query, font=fq)
    pw = lw + qw + 40; x = 40
    d.rounded_rectangle([x, y, x+pw, y+34], 8, fill=(22, 28, 36), outline=(40, 50, 60), width=1)
    d.text((x+16, y+9), lab, fill=accent, font=fl)
    d.text((x+16+lw, y+8), query, fill=INK, font=fq)
    return y + 34


def frame(W, H, orient, spec, panebase, scales, w_elapsed, dilate):
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    fh = font(max(22, W//44)); ft = font(16, False)
    a, b = spec["engines"]
    d.text((40, 26), a["label"], fill=a["_c"], font=fh)
    x = 40 + d.textlength(a["label"], font=fh)
    d.text((x+12, 30), "vs", fill=DIM, font=ft)
    d.text((x+44, 26), b["label"], fill=INK, font=fh)
    note = spec.get("note", "")
    if note: d.text((W-40-d.textlength(note, font=ft), 32), note, fill=DIM, font=ft)
    d.line([40, 74, W-40, 74], fill=(34, 43, 52), width=1)
    top = 96
    if spec.get("query"):
        query_bar(cv, W, 86, spec["query"], a["_c"]); top = 138
    t_real = w_elapsed / dilate
    rects = panes(W, H, orient, top)
    fastest = min(spec["engines"], key=lambda e: e["proc_s"])
    for r, e, base, sc in zip(rects, spec["engines"], panebase, scales):
        draw_pane(cv, r, base, sc, spec, e, min(1.0, t_real/e["proc_s"]), t_real, e is fastest)
    return cv


def end_card(W, H, spec):
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    a, b = sorted(spec["engines"], key=lambda e: e["proc_s"])
    ratio = b["proc_s"]/a["proc_s"]
    cx, top = int(W*0.16), int(H*0.28); fs = font(20); fl = font(15, False); big = font(max(34, W//30))
    d.text((cx, top), spec.get("note", "identical output").upper(), fill=DIM, font=fs)
    wbar = int(W*0.42)
    for i, e in enumerate(spec["engines"]):
        y = top + 56 + i*64; d.text((cx, y), e["label"], fill=e["_c"], font=fs)
        bl = int(wbar * (a["proc_s"]/e["proc_s"]))
        d.rectangle([cx+int(W*0.30), y+6, cx+int(W*0.30)+bl, y+14], fill=e["_c"])
        d.text((cx+int(W*0.30)+wbar+16, y), f"{e['proc_s']:.1f} s", fill=INK, font=fs)
        d.text((cx, y+24), e.get("rate", ""), fill=DIM, font=fl)
    d.text((cx, top+200), f"{ratio:.1f}× faster", fill=ACCENTS["teal"], font=big)
    if spec.get("link"): d.text((cx, top+280), spec["link"], fill=DIM, font=fl)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec"); ap.add_argument("--out", default=str(HERE / "out" / "image_race.mp4"))
    ap.add_argument("--layout", choices=list(LAYOUTS), default="cols")
    ap.add_argument("--dilate", type=float, default=0.0, help="playback factor; 0 = auto-fit to ~11 s")
    ap.add_argument("--gif", action="store_true", help="also write a .gif next to the mp4")
    ap.add_argument("--no-card", action="store_true", help="omit the end card (for concatenating scenes)")
    a = ap.parse_args()
    W, H, orient = LAYOUTS[a.layout]
    spec, img = load_spec(a.spec)
    top = 138 if spec.get("query") else 96
    rects = panes(W, H, orient, top)
    panebase, scales = [], []
    for r in rects:
        b, s = fit(img, r[2]-20, r[3]-90); panebase.append(b); scales.append(s)
    proc_max = max(e["proc_s"] for e in spec["engines"])
    dilate = a.dilate if a.dilate > 0 else max(0.02, 11.0 / proc_max)
    wall = proc_max * dilate
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); k = 0
        for i in range(int(wall*FPS)+1):
            frame(W, H, orient, spec, panebase, scales, i/FPS, dilate).save(tmp/f"f{k:05d}.png"); k += 1
        last = frame(W, H, orient, spec, panebase, scales, wall, dilate)
        for _ in range(int(0.8*FPS)): last.save(tmp/f"f{k:05d}.png"); k += 1
        if not a.no_card:
            card = end_card(W, H, spec)
            for _ in range(int(3.2*FPS)): card.save(tmp/f"f{k:05d}.png"); k += 1
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(tmp/"f%05d.png"), "-pix_fmt", "yuv420p", str(out)], check=True)
        if a.gif:
            pal = tmp/"pal.png"; gw = 760 if orient == "h" else 600
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                            "-vf", f"fps=13,scale={gw}:-1:flags=lanczos,palettegen=stats_mode=diff", str(pal)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(pal),
                            "-lavfi", f"fps=13,scale={gw}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                            str(out.with_suffix(".gif"))], check=True)
    print("wrote", out, ("+ gif" if a.gif else ""))


if __name__ == "__main__":
    main()
