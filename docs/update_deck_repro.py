"""Add our reproduced (sr_vggt) results to SimpleSR-project-update.pptx.

Reads docs/SimpleSR-project-update.before-repro.pptx (the deck as received) and writes
docs/SimpleSR-project-update.pptx:
- slide 4 and slide 9 tables: a new "Ours" column next to the deck's numbers, plus a one-line note;
- a new slide 10 (before the status slide): the paper's Table 2 (scene language, zero-shot) next to our rerun;
- slide numbers shifted after it; status slide: reproduction items added.
usage: python docs/update_deck_repro.py   (needs python-pptx)
"""
import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.util import Emu, Pt

D = Path(__file__).resolve().parent
SRC, DST = D / "SimpleSR-project-update.before-repro.pptx", D / "SimpleSR-project-update.pptx"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
INK, AMBER = "1F2937", "B45309"

# ---- our numbers (sources: docs/reproduce_exps.md, data/scenelang_vsibench_scannetpp/, results/) ----
OURS_SLIDE4 = ["Ours", "36.5", "52.3", "training"]          # zero-shot 2B; novggt; video+scene SFT (sr-rvs) running
OURS_SLIDE9 = ["Ours", "50.9", "60.2", "59.9", "–", "–", "–", "–", "–", "–", "–"]  # sr-main = row 3 (main)
NOTE4 = "Ours: sr_vggt rerun (frames decoded live at 1 fps, ≤ 64). Scene-language SFT on reason-190k is training."
NOTE9 = "Ours: sr_vggt rerun on VSI-Bench full; frames decoded live, VGGT in bf16. Perceiver rerun: 59.9 at step 743."
Q4B = {}  # Qwen3-VL-4B measurements (8-step smoke run)
NOVGGT_FULL = "45.5 68.3 48.1 95.4 51.9 35.0 35.9 9.1 20.3".split()  # 2B SFT, scene, full prompt (novggt_2b_scene_full)
RUNNING = "running – – – – – – – –"
ZS2B = {k: RUNNING for k in ("scene_basic", "scene_full", "video_scene")}  # zero-shot Qwen3-VL-2B, job 1363803

TASKS = ["Avg.", "Count", "Abs. dist", "Size", "Room", "Rel. dist", "Rel. dir", "Route", "Appear."]
T2 = [  # (label, kind, 9 numbers); kind: head = paper row, ours, ours_new (highlight), group, oracle
    ("Qwen3.5-4B, zero-shot", "group", None),
    ("Video · paper", "paper", "40.4 34.4 27.8 51.5 28.9 61.2 36.5 18.2 64.5"),
    ("Video · ours", "ours", "42.9 33.4 24.1 56.9 46.2 50.5 35.9 27.3 68.8"),
    ("Scene · paper", "paper", "55.4 75.2 66.6 46.4 48.6 70.9 68.9 27.3 39.1"),
    ("Scene · ours, basic prompt", "ours", "56.7 95.6 39.1 88.9 31.4 68.0 58.6 36.4 35.5"),
    ("Scene · ours, full prompt", "best", "69.2 93.6 81.1 100.0 42.7 79.6 95.2 22.7 38.4"),
    ("Video + scene · paper", "paper", "44.6 77.4 19.2 71.7 25.7 60.2 37.0 9.1 56.5"),
    ("Video + scene · ours", "ours", "51.0 91.3 33.2 74.6 40.8 46.6 36.5 22.7 62.3"),
    ("Qwen3-VL-2B, zero-shot", "group", None),
    ("Video · paper (Table 1, avg. only)", "paper", "38.0 – – – – – – – –"),
    ("Video · ours", "ours", "36.5 46.1 29.8 54.1 29.7 42.7 31.3 0.0 58.0"),
    ("Qwen3-VL-2B + spatial SFT (trained on video)", "group", None),
    ("Video · paper (Table 1, avg. only)", "paper", "47.8 – – – – – – – –"),
    ("Video · ours", "ours", "52.3 55.1 36.3 56.7 61.6 52.4 34.2 50.0 72.5"),
    ("Scene · ours, basic prompt", "ours", "43.7 77.4 22.3 60.2 28.6 30.1 29.5 54.5 47.1"),
    ("Scene · ours, full prompt", "ours", " ".join(NOVGGT_FULL) if NOVGGT_FULL else RUNNING),
    ("No model: computed from the boxes", "oracle", "93.7* 98.9 94.8 100.0 68.6 100.0 100.0 – –"),
]
VSI_ROWS = [  # VSI-Bench full split, ours (docs/reproduce_exps.md; results/sr_vggt_novggt/eval_checkpoint-743_full)
    ("Zero-shot (no adapter, no system prompt)", "group", None),
    ("Qwen3-VL-2B", "ours", "50.9 63.3 42.4 71.0 50.1 47.6 42.3 28.9 61.5"),
    ("Qwen3.5-4B", "ours", "55.1 52.3 39.2 67.5 57.3 57.9 61.4 36.6 68.8"),
    ("Fine-tuned Qwen3-VL-2B (sr_vggt, 190k mix, LoRA)", "group", None),
    ("SFT only, no VGGT (novggt)", "best", "60.2 71.3 43.9 74.0 69.0 61.1 47.7 38.7 76.1"),
    ("Perceiver + VGGT-Ω (main)", "ours", "59.9 70.9 45.1 74.0 68.5 60.3 48.6 34.0 78.0"),
]
REVSI_ROWS = [  # playground/zeroshot_revsi/*/results_merged.json; ReVSI has no appearance-order task
    ("Zero-shot, 1 fps sampling (as our VSI-Bench eval)", "group", None),
    ("Qwen3-VL-2B", "ours", "44.1 37.9 44.6 64.7 41.6 41.4 33.2 45.2 –"),
    ("Qwen3.5-4B", "ours", "45.2 31.8 50.7 59.1 48.4 48.7 43.6 34.4 –"),
    ("Zero-shot, ReVSI protocol (all 64 provided frames)", "group", None),
    ("Qwen3-VL-2B", "ours", "44.4 37.9 46.4 65.1 40.1 41.6 33.0 46.6 –"),
    ("Qwen3.5-4B", "best", "45.3 31.9 51.4 59.6 47.5 49.4 43.5 34.1 –"),
]
VR_BULLETS = [
    "Zero-shot: base models unchanged; frames at 1 fps (≤ 64), 64–128 tokens per frame; lmms-eval prompts and "
    "scoring; greedy, up to 16 new tokens.",
    "Fine-tuning: Qwen3-VL-2B, LoRA r = 64 on the LLM, 190k mix, 743 steps at batch 256 on 2× H100; frames decoded "
    "live. The Perceiver + VGGT-Ω run (VGGT in bf16) reaches 59.9 (reference 60.0).",
    "ReVSI: official lmms-eval prompts and scoring, 64-frame set, 380 videos. Under 1 fps, 152 videos get fewer "
    "than 64 frames; the ReVSI protocol feeds all 64. Both protocols agree within 0.3 points.",
]
COST_HEAD = ["SFT, 2× H100, 1 epoch", "Qwen3-VL-2B", "+ Perceiver", "Qwen3-VL-4B"]
COST_ROWS = [  # 2B: sr_vggt_novggt logs; Perceiver (2B + VGGT-Ω): sr_vggt_main logs (14 links, 743 steps); 4B: 8-step run
    ("Trainable params", "69.7 M (3.2%)", "373 M (10.1%)", Q4B.get("params", "measuring")),
    ("Peak memory per GPU", "16.8 GiB", "29.3 GiB", Q4B.get("mem", "measuring")),
    ("Time per step (256 Qs)", "73.4 s", "116.5 s", Q4B.get("step", "measuring")),
    ("Time per question", "0.57 GPU-s", "0.91 GPU-s", Q4B.get("q", "measuring")),
    ("Total GPU-hours", "31.3", "51.2", Q4B.get("total", "measuring")),
    ("Wall-clock", "15.6 h", "25.6 h", Q4B.get("wall", "measuring")),
]
RECIPE = [
    "Data: 190k QA (116k VSI-590K + 74k VLM-3R), 4,878 videos, 1 epoch",
    "Video: 1 fps, ≤ 64 frames, 64–128 tokens/frame, decoded live",
    "LoRA r = 64, α = 128, dropout 0.1 on every LLM linear layer; ViT and LLM frozen",
    "AdamW, lr 1e-4, cosine, 3% warmup, wd 0.01, bf16",
    "Global batch 256 = 1 × 128 accumulation × 2 GPUs",
    "SDPA attention, Liger fused CE, no gradient checkpointing",
    "Perceiver run: + 512 frame / 32 scene latents over frozen VGGT-Ω, run live in bf16",
]
COST_NOTE = ("+ Perceiver = Qwen3-VL-2B + Perceiver over frozen VGGT-Ω (main). GPU-h = wall-clock × 2 GPUs, start-ups "
             "included. Both 2B runs: measured over the full 743 steps. "
             "4B: 8-step run, extrapolated.")
BULLETS = [
    "752 debiased ScanNet++ questions (Table 2's split). Scene text from VSI-Bench's metadata: one 9-DoF box "
    "per object, z up. Greedy decoding, thinking off.",
    "Basic prompt: one line on the box format. Full prompt: every field plus each task's computation method; "
    "outputs up to 16,384 tokens (4B) or 4,096 (2B, longer outputs only loop).",
    "Scene text ≫ video, as in the paper. With the basic prompt the SFT model answers in one token, without computing. "
    "* 6 tasks: no time order, no route.",
]


def runs(cell):
    return cell._tc.findall(f".//{A}r")


def set_cell(cell, text, color=INK, bold=None):
    """Keep the cell's first run (font, size, colour) and replace its text."""
    rs = runs(cell)
    for r in rs[1:]:
        r.getparent().remove(r)
    r = rs[0]
    r.find(f"{A}t").text = text
    rpr = r.find(f"{A}rPr")
    rpr.find(f".//{A}srgbClr").set("val", color)
    if bold is not None:
        rpr.set("b", "1" if bold else "0")


def add_column(tbl, width):
    """Append a column, each new cell copied from the row's last cell."""
    grid = tbl._tbl.find(f"{A}tblGrid")
    gc = copy.deepcopy(grid[-1]); gc.set("w", str(width)); grid.append(gc)
    for tr in tbl._tbl.findall(f"{A}tr"):
        tr.append(copy.deepcopy(tr.findall(f"{A}tc")[-1]))


def set_widths(tbl, widths):
    for gc, w in zip(tbl._tbl.find(f"{A}tblGrid"), widths):
        gc.set("w", str(w))


def find(slide, text):
    return next(s for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip().startswith(text))


def table(slide):
    return next(s for s in slide.shapes if s.has_table)


def note(slide, like, text, top, left, width, size=8):
    """A one-paragraph text box styled like `like` (a deck text box), at the given place."""
    el = copy.deepcopy(like._element)
    slide.shapes._spTree.append(el)
    sh = slide.shapes[-1]
    sh.left, sh.top, sh.width, sh.height = Emu(left), Emu(top), Emu(width), Emu(Pt(size * 2.6))
    p0 = sh.text_frame.paragraphs[0]
    for p in sh.text_frame.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    for r in p0.runs[1:]:
        r._r.getparent().remove(r._r)
    p0.runs[0].text = text
    p0.runs[0].font.size = Pt(size)
    p0.runs[0].font.italic = True
    return sh


def set_bullet(para, text):
    """A deck bullet is two runs: a bold teal '•  ' and the regular text. Keep both, replace the text."""
    rs = para.runs
    for r in rs[2:]:
        r._r.getparent().remove(r._r)
    rs[1].text = text


def bullets(shape, items):
    """Append bullet paragraphs, each a copy of the shape's last paragraph."""
    for text in items:
        shape.text_frame._txBody.append(copy.deepcopy(shape.text_frame.paragraphs[-1]._p))
        set_bullet(shape.text_frame.paragraphs[-1], text)


prs = Presentation(SRC)
S = prs.slides

# ---- slide 4: Qwen3-VL-2B debiased ScanNet++ table ----
s4 = S[3]; t4 = table(s4)
add_column(t4.table, 750000)
set_widths(t4.table, [2950000, 650000, 550000, 700000])
for i, (row, val) in enumerate(zip(t4.table.rows, OURS_SLIDE4)):
    set_cell(list(row.cells)[-1], val, "FFFFFF" if i == 0 else (AMBER if val == "training" else INK), bold=i == 0)
note(s4, find(s4, "Scene language: one JSON"), NOTE4, top=4560000, left=411491, width=5000000, size=7)

# ---- slide 9: VSI-Bench full table ----
s9 = S[8]; t9 = table(s9)
add_column(t9.table, 700000)
set_widths(t9.table, [2750000, 800000, 700000, 750000])
for i, (row, val) in enumerate(zip(t9.table.rows, OURS_SLIDE9)):
    c = list(row.cells)[-1]
    if i == 0:
        set_cell(c, val, "FFFFFF", bold=True)
    else:
        set_cell(c, val, AMBER if val == "training" else INK, bold=False)
note(s9, find(s9, "SimpleSR · work"), NOTE9, top=4380000, left=411491, width=5300000, size=7)

# ---- new slides, built from a copy of slide 9's frame (kicker, title, footer, page number, table) ----
FILL = {"paper": "F3F4F6", "na": "F3F4F6", "ours": "FFFFFF", "best": "E6F4F2", "group": "E5E7EB", "oracle": "FEF3E2"}


def new_slide(kicker, title):
    sl = prs.slides.add_slide(s9.slide_layout)
    for el in s9.shapes._spTree.iterchildren():
        if el.tag.endswith(("}sp", "}graphicFrame")):
            sl.shapes._spTree.append(copy.deepcopy(el))
    find(sl, "PRELIMINARY").text_frame.paragraphs[0].runs[0].text = kicker
    find(sl, "VSI-Bench: a 2B").text_frame.paragraphs[0].runs[0].text = title
    for sh in list(sl.shapes):  # drop slide 9's takeaways panel and its note
        if sh.has_text_frame and sh.text_frame.text.startswith(("Takeaways", "•  A 2B", "Ours: sr_vggt")) or \
                (sh.shape_type == 1 and sh.left > 5000000 and sh.top < 2000000):
            sh._element.getparent().remove(sh._element)
    return sl


def fill_table(gf, header, rows, top, row_h=160000, head_h=170000, sz="800"):
    """Reshape a copy of slide 9's table to 1 + len(TASKS) columns and the given rows (label, kind, numbers)."""
    tb = gf.table
    while len(tb._tbl.find(f"{A}tblGrid")) < 1 + len(TASKS):
        add_column(tb, 600000)
    while len(tb._tbl.find(f"{A}tblGrid")) > 1 + len(TASKS):
        g = tb._tbl.find(f"{A}tblGrid"); g.remove(g[-1])
        for tr in tb._tbl.findall(f"{A}tr"):
            tr.remove(tr.findall(f"{A}tc")[-1])
    set_widths(tb, [2900000] + [600000] * len(TASKS))
    trs = tb._tbl.findall(f"{A}tr")
    body = copy.deepcopy(trs[1])
    for tr in trs[1:]:
        tb._tbl.remove(tr)
    for label, kind, nums in rows:
        tr = copy.deepcopy(body); tb._tbl.append(tr)
        tr.set("h", str(row_h))
        for tc in tr.findall(f"{A}tc"):
            tc.find(f".//{A}tcPr/{A}solidFill/{A}srgbClr").set("val", FILL[kind])
    for c, h in zip(tb.rows[0].cells, [header] + TASKS):
        set_cell(c, h, "FFFFFF", bold=True)
    tb._tbl.findall(f"{A}tr")[0].set("h", str(head_h))
    for row, (label, kind, nums) in zip(list(tb.rows)[1:], rows):
        vals = nums.split() if nums and kind != "na" else [""] * len(TASKS)
        set_cell(row.cells[0], label, INK, bold=kind == "group")
        for j, (c, v) in enumerate(zip(list(row.cells)[1:], vals)):
            set_cell(c, v, AMBER if v in ("running", "training") else INK, bold=kind == "best" and j == 0)
        if kind == "na":  # one merged, greyed note across the score cells
            cells = list(row.cells)
            cells[1].merge(cells[-1])
            set_cell(cells[1], nums, "6B7280", bold=False)
            cells[1].text_frame.paragraphs[0]._p.find(f"{A}pPr").set("algn", "l")
        if kind == "group":  # one merged label cell across the row
            row.cells[0].merge(list(row.cells)[-1])
            set_cell(row.cells[0], label, INK, bold=True)
    for row in tb.rows:
        for c in row.cells:
            for r in runs(c):
                r.find(f"{A}rPr").set("sz", sz)
    gf.top = Emu(top)


def notes_box(sl, items, top, height, size=7):
    """Bullets styled like slide 4's 'Why it works' list."""
    sl.shapes._spTree.append(copy.deepcopy(find(S[3], "•  Structured text")._element))
    bb = sl.shapes[-1]
    bb.left, bb.top, bb.width, bb.height = Emu(411491), Emu(top), Emu(8330000), Emu(height)
    paras = bb.text_frame.paragraphs
    for p_ in paras[1:]:
        p_._p.getparent().remove(p_._p)
    set_bullet(paras[0], items[0])
    bullets(bb, items[1:])
    for p_ in bb.text_frame.paragraphs:
        for r in p_.runs:
            r.font.size = Pt(size)


# slide 10: Table 2 reproduction
t2 = new_slide("REPRODUCTION · FINDING 2", "Scene language, zero-shot: Table 2 vs. our rerun")
fill_table(table(t2), "Qwen, debiased ScanNet++ (752 Qs)", T2, top=950000)
notes_box(t2, BULLETS, top=3900000, height=700000)

# slide 11: VSI-Bench full split and ReVSI
vr = new_slide("REPRODUCTION · BASELINES", "Our reruns on VSI-Bench and ReVSI")
vr.shapes._spTree.append(copy.deepcopy(table(vr)._element))  # second table (ReVSI), copied while still unfilled
vsi_gf, revsi_gf = [s for s in vr.shapes if s.has_table]
fill_table(vsi_gf, "VSI-Bench, full split (5,130 Qs)", VSI_ROWS, top=950000, row_h=165000, head_h=185000)
fill_table(revsi_gf, "ReVSI, 64-frame set (6,616 Qs)", REVSI_ROWS, top=2300000, row_h=165000, head_h=185000)
notes_box(vr, VR_BULLETS, top=3650000, height=950000, size=7.5)

# slide 12: SFT recipe and cost (slide 9's frame: table left, panel right)
cs = prs.slides.add_slide(s9.slide_layout)
for el in s9.shapes._spTree.iterchildren():
    if el.tag.endswith(("}sp", "}graphicFrame")):
        cs.shapes._spTree.append(copy.deepcopy(el))
find(cs, "PRELIMINARY").text_frame.paragraphs[0].runs[0].text = "REPRODUCTION · TRAINING COST"
find(cs, "VSI-Bench: a 2B").text_frame.paragraphs[0].runs[0].text = "SFT recipe and training cost"
for sh in list(cs.shapes):
    if sh.has_text_frame and sh.text_frame.text.startswith("Ours: sr_vggt"):
        sh._element.getparent().remove(sh._element)
find(cs, "Takeaways").text_frame.paragraphs[0].runs[0].text = "SFT recipe"
rb = find(cs, "•  A 2B")
for p_ in rb.text_frame.paragraphs[1:]:
    p_._p.getparent().remove(p_._p)
set_bullet(rb.text_frame.paragraphs[0], RECIPE[0])
bullets(rb, RECIPE[1:])
for p_ in rb.text_frame.paragraphs:
    for r in p_.runs:
        r.font.size = Pt(9)
ct = table(cs).table
for tr in ct._tbl.findall(f"{A}tr")[1 + len(COST_ROWS):]:
    ct._tbl.remove(tr)
g = ct._tbl.find(f"{A}tblGrid")
while len(g) > 4:  # keep 4 columns (slide 9's table has exactly 4)
    g.remove(g[-1])
    for tr in ct._tbl.findall(f"{A}tr"):
        tr.remove(tr.findall(f"{A}tc")[-1])
set_widths(ct, [1700000, 1150000, 1200000, 1150000])
for c, h in zip(ct.rows[0].cells, COST_HEAD):
    set_cell(c, h, "FFFFFF", bold=True)
ct._tbl.findall(f"{A}tr")[0].set("h", "330000")
for row, vals in zip(list(ct.rows)[1:], COST_ROWS):
    row._tr.set("h", "330000")
    for j, (c, v) in enumerate(zip(row.cells, vals)):
        set_cell(c, v, AMBER if v == "measuring" else INK, bold=False)
        c._tc.find(f".//{A}tcPr/{A}solidFill/{A}srgbClr").set("val", "F3F4F6" if j == 0 else "FFFFFF")
for row in ct.rows:
    for c in row.cells:
        for r in runs(c):
            r.find(f"{A}rPr").set("sz", "900")
note(cs, find(cs, "SimpleSR · work"), COST_NOTE, top=3550000, left=411491, width=5200000, size=7)
cs.shapes[-1].height = Emu(420000)

# slide 11: the two scene-language prompts, in slide 7's code panels
import ast, json
EV = D.parent / "data" / "scenelang_vsibench_scannetpp"
_tree = ast.parse((EV / "eval_scenelang.py").read_text())
PR = {t.id: ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
      for t in n.targets if isinstance(t, ast.Name) and t.id in ("SCENE_INTRO", "FULL_INTRO", "HOW")}
ex = next(r for r in map(json.loads, open(EV / "vsibench_scannetpp_scenelang.jsonl"))
          if r["debiased"] and r["question_type"].startswith("object_rel_direction"))
lines = ex["scene_language"].split("\n")
scene = lines[:3] + ["  ... one line per object ..."] + lines[-1:]
q = [ex["question"].split("\n")[0], "Options: " + "  ".join(ex["options"])]
BASIC = [("", PR["SCENE_INTRO"]), ("", "<scene>"), *[("", l) for l in scene], ("", "</scene>"), *[("", l) for l in q],
         ("new", "Answer with the option's letter from the given choices directly.")]
how = next(h for k, h in PR["HOW"].items() if ex["question_type"].startswith(k))
FULL = [*[("new", l) for l in PR["FULL_INTRO"].split("\n")], ("", "<scene>"), *[("", l) for l in scene], ("", "</scene>"),
        *[("", l) for l in q], ("new", how),
        ("new", "Compute the answer step by step from the numbers above, then end with a final line 'Answer: <option letter>'.")]
s7 = S[6]
panel = next(sh for sh in s7.shapes if sh.shape_type == 1 and sh.width > 4000000)
code = find(s7, "Q: Standing")
ps = new_slide("REPRODUCTION · FINDING 2", "Scene-language prompts: basic vs. full")
for sh in list(ps.shapes):  # keep only kicker, title, footer and page number
    if sh.has_table:
        sh._element.getparent().remove(sh._element)
for k, (title, sub, body, left) in enumerate([
        ("Basic prompt", "Qwen3.5-4B 56.7 · Rel. dir 58.6", BASIC, 411491),
        ("Full prompt", "Qwen3.5-4B 69.2 · Rel. dir 95.2", FULL, 4640000)]):
    w = 4100000
    ps.shapes._spTree.append(copy.deepcopy(panel._element)); pn = ps.shapes[-1]
    pn.left, pn.top, pn.width, pn.height = Emu(left), Emu(1250000), Emu(w), Emu(3200000)
    ps.shapes._spTree.append(copy.deepcopy(code._element)); cb = ps.shapes[-1]
    cb.left, cb.top, cb.width, cb.height = Emu(left + 70000), Emu(1300000), Emu(w - 140000), Emu(3100000)
    tf = cb.text_frame
    for p_ in tf.paragraphs[1:]:
        p_._p.getparent().remove(p_._p)
    first = tf.paragraphs[0]
    for r in first.runs[1:]:
        r._r.getparent().remove(r._r)
    for i, (tag, text) in enumerate(body):
        para = first if i == 0 else None
        if para is None:
            tf._txBody.append(copy.deepcopy(first._p)); para = tf.paragraphs[-1]
        para.runs[0].text = text
        para.runs[0].font.size = Pt(6)
        para.runs[0]._r.find(f".//{A}srgbClr").set("val", "5EEAD4" if tag == "new" else "E5E7EB")
    note(ps, find(ps, "SimpleSR · work"), f"{title} — {sub}", top=1000000, left=left, width=w, size=10)
    hb = ps.shapes[-1]; r0 = hb.text_frame.paragraphs[0].runs[0]
    r0.font.italic = False; r0.font.bold = True; r0._r.find(f".//{A}srgbClr").set("val", INK)
note(ps, find(ps, "SimpleSR · work"),
     "A debiased relative-direction question; scene shortened (the model sees every object). "
     "Teal: what differs between the prompts.",
     top=4500000, left=411491, width=8330000, size=7)

# move the new slides before the status slide and renumber the page numbers
ids = prs.slides._sldIdLst
moved = [ids[-4], ids[-1], ids[-3], ids[-2]]  # Table 2, prompts, VSI-Bench / ReVSI, recipe & cost
for el in moved:
    ids.remove(el)
for k, el in enumerate(moved):
    ids.insert(9 + k, el)
for k, slide in enumerate(prs.slides, 1):
    num = next(s for s in slide.shapes if s.has_text_frame and s.left > 8000000 and s.top > 4700000
               and s.text_frame.text.strip().isdigit()) if k > 1 else None
    if num is not None:
        num.text_frame.paragraphs[0].runs[0].text = str(k)

# ---- status slide ----
st = prs.slides[13]
bullets(find(st, "•  Analysis"), ["Reproduced: VSI-Bench, ReVSI, Table 2, Perceiver run (59.9)"])
run_box = find(st, "•  Full SimpleSR")
bullets(run_box, ["SFT on video + scene language (reason-190k)", "SFT of Qwen3.5-4B, no VGGT (190k mix)"])
run_box.height = find(st, "•  Analysis").height
prs.save(DST)
print("wrote", DST)
