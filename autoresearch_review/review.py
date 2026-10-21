"""Independent code review by a local LLM (Qwen via Ollama) — 19/10 sera.

Three lenses on every chunk of the pipeline code (code_snapshot/):
  leakage    : information from the scored year / future / test / holdout entering a fit or a feature
  bugs       : correctness errors (joins, units, years, exposure, NaN, signs, row order, categories)
  part2      : assumptions hard-coded to the Part 1 market that would break or mislead on a new market
Then a second pass: every high/medium finding goes back to the model with the same chunk, asking it to
refute the claim (REAL / FALSE_POSITIVE / UNSURE). Output: findings.jsonl, reviews/<file>.md, SINTESI.md.
The actuary verifies the REAL/UNSURE ones in the container before acting on any.

Resume-safe (state in done.json). Stop: touch autoresearch_review/STOP.
Usage: python -I autoresearch_review/review.py [--host http://localhost:11434] [--model auto] [--mock]
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code_snapshot"
OUT = HERE / "reviews"
FIND = HERE / "findings.jsonl"
DONE = HERE / "done.json"
STOP = HERE / "STOP"
PREFERRED = ["qwen3.8:27b", "qwen3.8", "qwen3.6:35b-a3b", "qwen3.6", "qwen3.5", "qwen3:32b", "qwen3:30b-a3b", "qwen3"]
# order = pipeline order; models_extra is split by function
FILES = ["prepare.py", "market_config.py", "clean_policies.py", "clean_claims.py", "develop.py", "trend.py",
         "build_datasets.py", "features.py", "experience.py", "models_extra.py", "oof.py", "blend.py",
         "holdout_weights.py", "final.py", "final_line.py", "metrics.py", "transfer.py", "audit_new_market.py",
         "run_all.sh", "parte1_s4.json"]
MAX_LINES = 220

LENSES = {
    "leakage": """Look ONLY for data leakage / temporal validity problems: anything that lets information from the
scored year, a later year, the 2025 test rows or the holdout policies enter a fitted model, a feature, an encoding,
a credibility estimate, a calibration or a weight. Also: folds that are not really out-of-time, features computed
including the current period's claims, statistics fit on train+valid together.""",
    "bugs": """Look ONLY for correctness bugs: wrong joins/merges (duplicates, dropped rows, misaligned order after
a merge or groupby), unit or scale errors, off-by-one years, exposure or offset mistakes (log vs level), wrong sign or
formula (deviance, credibility, trend projection (1+t)^(2025-year)), NaN silently becoming 0 or a category,
train/valid category levels not aligned, integer division, a variable shadowed or reused by mistake, results
depending on row order, a parameter that is read but never used.""",
    "part2": """Look ONLY for things that would break or silently mislead when the SAME code runs on a NEW market
(Part 2): hard-coded years (2021-2025), hard-coded perils/lines/column names/regions, thresholds tuned to Part 1
sizes (claim counts, min_data, caps in currency units), assumptions on units (sum insured, currency), code paths
that assume a column exists, silent fallbacks that would hide a problem.""",
}

FORMAT = """Answer with a JSON array only (no prose before or after). Each element:
{"line_from": int, "line_to": int, "severity": "high"|"medium"|"low", "claim": "<one sentence: what is wrong>",
 "why": "<one or two sentences: concrete consequence on predictions or validation>",
 "quote": "<the exact code line(s) involved, copied>", "test": "<a quick check a human can run to confirm>"}
Report only concrete issues you can point to in THIS code, with the quoted line. Do not report style, missing
docstrings, performance, or generic advice. If you find nothing real, answer []. At most 6 elements."""


def chunks_of(path):
    lines = path.read_text().splitlines()
    if path.suffix != ".py" or len(lines) <= MAX_LINES:
        return [(1, len(lines))], lines
    # greedy: cut only at top-level boundaries (def/class/constant/decorator), chunks of at most MAX_LINES
    bounds = sorted({0, len(lines)} | {i for i, l in enumerate(lines) if re.match(r"^(def |class |[A-Z_]+ = |@)", l)})
    out, a, last = [], 0, 0
    for s in bounds[1:]:
        if s - a > MAX_LINES and last > a:
            out.append((a + 1, last)); a = last
        while s - a > MAX_LINES:              # a single function longer than MAX_LINES
            out.append((a + 1, a + MAX_LINES)); a += MAX_LINES
        last = s
    if a < len(lines):
        out.append((a + 1, len(lines)))
    return out, lines


def numbered(lines, a, b):
    return "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1))


def header_of(lines):
    h = []
    for i, l in enumerate(lines[:60], 1):
        if re.match(r"^(import |from |[A-Z_]+ ?=|\"\"\"|#)", l) or (h and i <= 25):
            h.append(f"{i:4d}| {l}")
    return "\n".join(h[:45])


# ---------------------------------------------------------------- LLM
def pick_model(host, wanted):
    if wanted != "auto":
        return wanted
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
        avail = [m["name"] for m in json.load(r).get("models", [])]
    for pref in PREFERRED:
        for name in avail:
            if name == pref or name.startswith(pref + ":") or name.split(":")[0] == pref:
                return name
    if avail:
        return avail[0]
    raise SystemExit("Ollama has no models")


def ask(host, model, prompt, mock=False, timeout=1500):
    if mock:
        return '[{"line_from": 1, "line_to": 1, "severity": "medium", "claim": "mock", "why": "mock", "quote": "x", "test": "y"}]' \
            if "JSON array" in prompt else "VERDICT: UNSURE\nREASON: mock"
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": "You are a meticulous senior actuary and Python reviewer. "
                          "You only report issues you can prove from the code shown."},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "num_ctx": 32768, "num_predict": 4096}}
    for attempt in range(2):
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = json.load(r)["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
        except urllib.error.HTTPError as e:
            if attempt == 0 and "think" in body:
                body.pop("think"); continue
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read()[:200]}")
    return ""


def parse_findings(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return [x for x in arr if isinstance(x, dict) and x.get("claim")]
    except json.JSONDecodeError:
        return None


VERIFY = """Below is a code chunk and a claimed issue found by another reviewer. Try hard to REFUTE it: read the
code, consider what the surrounding pipeline guarantees (context above), and decide.
Answer exactly in this format:
VERDICT: REAL | FALSE_POSITIVE | UNSURE
REASON: <two or three sentences, citing line numbers>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    model = "mock" if a.mock else pick_model(a.host, a.model)
    ctx = (HERE / "CONTEXT.md").read_text()
    OUT.mkdir(exist_ok=True)
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    print(f"model: {model}", flush=True)
    t_all = time.time()
    # ---------- pass 1: review
    for fname in FILES:
        path = CODE / fname
        if not path.exists():
            print(f"skip {fname} (missing)", flush=True); continue
        chunks, lines = chunks_of(path)
        head = header_of(lines) if len(chunks) > 1 else ""
        for (ca, cb) in chunks:
            for lens, instr in LENSES.items():
                key = f"{fname}:{ca}-{cb}:{lens}"
                if key in done:
                    continue
                if STOP.exists():
                    print("STOP found: exit", flush=True); return
                prompt = (f"{ctx}\n\n# File `{fname}`, lines {ca}-{cb} of {len(lines)}\n"
                          + (f"File header (context only):\n```\n{head}\n```\n" if head and ca > 1 else "")
                          + f"```\n{numbered(lines, ca, cb)}\n```\n\n# Your task\n{instr}\n\n{FORMAT}")
                t0 = time.time()
                try:
                    txt = ask(a.host, model, prompt, a.mock)
                except Exception as e:  # noqa: BLE001
                    print(f"[{key}] LLM error: {e}", flush=True); time.sleep(20); continue
                fs = parse_findings(txt)
                if fs is None:
                    fs = []; (OUT / "unparsed.txt").open("a").write(f"\n### {key}\n{txt}\n")
                with FIND.open("a") as f:
                    for x in fs:
                        x.update(file=fname, chunk=[ca, cb], lens=lens, verdict=None)
                        f.write(json.dumps(x) + "\n")
                done[key] = len(fs); DONE.write_text(json.dumps(done, indent=0))
                print(f"[{key}] {len(fs)} findings in {time.time() - t0:.0f}s", flush=True)
    # ---------- pass 2: try to refute every high/medium finding
    rows = [json.loads(l) for l in FIND.read_text().splitlines()] if FIND.exists() else []
    for i, x in enumerate(rows):
        if x.get("verdict") or x.get("severity") not in ("high", "medium"):
            continue
        if STOP.exists():
            print("STOP found: exit", flush=True); break
        lines = (CODE / x["file"]).read_text().splitlines()
        ca, cb = x["chunk"]
        prompt = (f"{ctx}\n\n# File `{x['file']}`, lines {ca}-{cb}\n```\n{numbered(lines, ca, cb)}\n```\n\n"
                  f"# Claimed issue ({x['lens']}, {x['severity']}), lines {x.get('line_from')}-{x.get('line_to')}\n"
                  f"Claim: {x['claim']}\nWhy: {x.get('why')}\nQuoted: {x.get('quote')}\n\n{VERIFY}")
        try:
            txt = ask(a.host, model, prompt, a.mock)
        except Exception as e:  # noqa: BLE001
            print(f"verify {i} LLM error: {e}", flush=True); continue
        m = re.search(r"VERDICT:\s*(REAL|FALSE_POSITIVE|UNSURE)", txt)
        x["verdict"] = m.group(1) if m else "UNSURE"
        r = re.search(r"REASON:\s*(.*)", txt, re.S)
        x["verdict_reason"] = (r.group(1).strip() if r else txt.strip())[:600]
        FIND.write_text("".join(json.dumps(y) + "\n" for y in rows))
        print(f"verify {i} {x['file']}:{x.get('line_from')} -> {x['verdict']}", flush=True)
    write_summary(rows, model, time.time() - t_all)


def write_summary(rows, model, secs):
    sev = {"high": 0, "medium": 1, "low": 2}
    ver = {"REAL": 0, "UNSURE": 1, None: 2, "FALSE_POSITIVE": 3}
    rows = sorted(rows, key=lambda x: (ver.get(x.get("verdict"), 2), sev.get(x.get("severity"), 3), x["file"]))
    L = [f"# Revisione del codice di Qwen ({model}) — sintesi", "",
         f"{len(rows)} segnalazioni; seconda passata (Qwen prova a smontarle) sulle high/medium. "
         "Da verificare a mano nel container prima di agire.", "",
         "| # | Verdetto | Gravità | Lente | File:righe | Segnalazione |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(rows, 1):
        L.append(f"| {i} | {x.get('verdict') or '–'} | {x.get('severity')} | {x['lens']} | "
                 f"{x['file']}:{x.get('line_from')}-{x.get('line_to')} | {str(x['claim']).replace('|', '/')} |")
    L += ["", "## Dettagli", ""]
    for i, x in enumerate(rows, 1):
        if x.get("verdict") == "FALSE_POSITIVE" or x.get("severity") == "low":
            continue
        L += [f"### {i}. {x['file']}:{x.get('line_from')}-{x.get('line_to')} ({x['lens']}, {x.get('severity')}, "
              f"{x.get('verdict')})", f"- **Cosa:** {x['claim']}", f"- **Perché conta:** {x.get('why')}",
              f"- **Codice:** `{str(x.get('quote'))[:300]}`", f"- **Test proposto:** {x.get('test')}",
              f"- **Seconda passata:** {x.get('verdict_reason', '–')}", ""]
    (HERE / "SINTESI.md").write_text("\n".join(L))
    print(f"SINTESI.md scritta ({len(rows)} segnalazioni, {secs / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
