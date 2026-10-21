"""Autoresearch loop (Karpathy-style) with a local LLM via Ollama.

Each iteration:
  1. ask the LLM for ONE change to experiment.py (prompt = program.md + best code + columns + recent results)
  2. static guardrails on the proposed code (imports, forbidden calls, contract)
  3. run the frozen judge eval_experiment.py in a subprocess with a time limit
  4. keep the change only if pooled_mean improves by more than --min-delta (git commit), else discard
  5. append a row to results.tsv
Stop: create the file autoresearch_freq/STOP, or reach --iters.
Usage:
  python -I autoresearch/run_loop.py --iters 200            # Ollama, model chosen automatically
  python -I autoresearch/run_loop.py --llm mock --iters 3   # test without LLM
"""
import argparse
import ast
import json
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE / "experiment.py"
RUNS = HERE / "runs"
RESULTS = HERE / "results.tsv"
STATE = HERE / "state.json"
COLUMNS = HERE / "columns.json"
STOP = HERE / "STOP"
PREFERRED = ["qwen3.6:35b-a3b", "qwen3.6", "qwen3.8:27b", "qwen3.8", "qwen3.5", "qwen3:30b-a3b", "qwen3:32b",
             "qwen3:14b", "qwen3-coder", "qwen2.5-coder:32b", "qwen3:8b", "qwen3"]
ALLOWED_IMPORTS = {"numpy", "pandas", "lightgbm", "sklearn", "scipy", "math", "itertools", "functools",
                   "collections", "re", "warnings", "os"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "breakpoint"}
FORBIDDEN_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "walk", "listdir", "chdir", "makedirs",
                   "read_parquet", "read_csv", "read_pickle", "to_parquet", "to_csv", "to_pickle", "read_excel",
                   "load", "save", "spawn", "fork", "kill"}


# ---------------------------------------------------------------- guardrails
def static_check(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import not allowed: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import not allowed: {node.module}"
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return f"forbidden name: {node.id}"
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return f"forbidden call: .{node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr != "environ":
                return "os may only be used for os.environ"
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "fit_predict" not in funcs or [a.arg for a in funcs["fit_predict"].args.args] not in (["train", "valid", "line"], ["train", "valid", "line", "peril"]):
        return "fit_predict(train, valid, line) missing or changed"
    if re.search(r"valid\s*\[\s*['\"]target", code):
        return "reads a target column of valid"
    return None


# ---------------------------------------------------------------- judge
def evaluate(path: Path, timeout: int, columns_out: Path | None = None):
    cmd = [sys.executable, "-I", str(HERE / "eval_experiment.py"), str(path)]
    if columns_out:
        cmd += ["--columns-out", str(columns_out)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s", time.time() - t0
    m = re.search(r"^RESULT (\{.*\})$", p.stdout, re.M)
    if p.returncode != 0 or not m:
        err = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["unknown error"]
        return None, f"run failed: {err[0][:300]}", time.time() - t0
    return json.loads(m.group(1)), "ok", time.time() - t0


# ---------------------------------------------------------------- LLM
def ollama_models(host):
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


def pick_model(host, wanted):
    avail = ollama_models(host)
    if wanted and wanted != "auto":
        return wanted
    for pref in PREFERRED:
        for name in avail:
            if name == pref or name.startswith(pref + ":") or name.split(":")[0] == pref:
                return name
    if avail:
        return avail[0]
    raise SystemExit("Ollama non ha modelli installati: esegui per esempio `ollama pull qwen3.6`")


def ask_ollama(host, model, prompt, timeout=1200):
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": "You are a careful actuarial data scientist. Follow the output format exactly."},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0.7, "num_ctx": 32768, "num_predict": 8192}}
    for attempt in range(2):
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = json.load(r)["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
        except urllib.error.HTTPError as e:
            if attempt == 0 and "think" in body:
                body.pop("think")          # vecchie versioni di Ollama non accettano "think"
                continue
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read()[:200]}")
    return ""


def ask_mock(best_code, it):
    """LLM finto per i test: cambia un iperparametro, ogni tanto propone codice vietato."""
    if it % 3 == 2:
        return "IDEA: read extra data\n```python\nimport subprocess\n" + best_code + "\n```"
    nl = random.choice([7, 15, 31]); lr = random.choice([0.02, 0.03, 0.05])
    code = re.sub(r"num_leaves=\d+", f"num_leaves={nl}", best_code)
    code = re.sub(r"learning_rate=[0-9.]+", f"learning_rate={lr}", code)
    return f"IDEA: num_leaves={nl}, learning_rate={lr}\n```python\n{code}\n```"


def parse_answer(txt):
    idea = re.search(r"IDEA:\s*(.+)", txt)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, flags=re.S)
    if not blocks:
        return None, None
    code = max(blocks, key=len)
    return (idea.group(1).strip()[:300] if idea else "(no idea line)"), code


def build_prompt(best_code, best, history):
    cols = json.loads(COLUMNS.read_text()) if COLUMNS.exists() else {}
    col_txt = "\n".join(f"- {line}: " + ", ".join(f"{c} ({t})" for c, t in d.items()) for line, d in cols.items())
    hist = "\n".join(f"- [{h['status']}] {h['score']} {h['idea']}" for h in history[-20:]) or "- (none yet)"
    return (f"{(HERE / 'program.md').read_text()}\n\n## Current best\npooled_mean = {best['pooled_mean']}, "
            f"per-line mean = {best['score']}, by fold: {json.dumps(best['by'])}\n\n## Columns available\n{col_txt}\n\n"
            f"## Recent attempts (newest last; do not repeat failed ideas)\n{hist}\n\n"
            f"## Current best experiment.py\n```python\n{best_code}\n```\n")


# ---------------------------------------------------------------- loop
def git(*args):
    try:
        subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, timeout=30)
    except Exception:
        pass


def log_row(row):
    new = not RESULTS.exists()
    with open(RESULTS, "a") as fh:
        if new:
            fh.write("iter\ttime\tmodel\tstatus\tpooled_mean\tscore\tdelta\tseconds\tidea\n")
        fh.write("\t".join(str(row.get(k, "")) for k in
                           ["iter", "time", "model", "status", "pooled_mean", "score", "delta", "seconds", "idea"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="ollama", choices=["ollama", "mock"])
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--min-delta", type=float, default=0.0003)
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    if not (HERE / ".git").exists():
        git("init"); git("add", "experiment.py", "program.md"); git("commit", "-m", "baseline")

    model = "mock" if a.llm == "mock" else pick_model(a.host, a.model)
    print(f"modello: {model}", flush=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "best" not in state:
        print("valuto la base...", flush=True)
        res, msg, sec = evaluate(EXP, a.timeout * 2, COLUMNS)
        if res is None:
            raise SystemExit(f"la base non gira: {msg}")
        state = {"best": res, "iter": 0, "history": []}
        STATE.write_text(json.dumps(state, indent=1))
        log_row({"iter": 0, "time": time.strftime("%Y-%m-%d %H:%M"), "model": model, "status": "baseline",
                 "pooled_mean": res["pooled_mean"], "score": res["score"], "delta": 0, "seconds": round(sec), "idea": "baseline"})
        print(f"base: pooled_mean={res['pooled_mean']} score={res['score']}", flush=True)

    start = state["iter"]
    for it in range(start + 1, start + a.iters + 1):
        if STOP.exists():
            print("STOP trovato: esco", flush=True); break
        best = state["best"]; best_code = EXP.read_text()
        t0 = time.time()
        try:
            answer = ask_mock(best_code, it) if a.llm == "mock" else ask_ollama(a.host, model, build_prompt(best_code, best, state["history"]))
        except Exception as e:
            print(f"[{it}] LLM errore: {e}", flush=True); time.sleep(30); continue
        idea, code = parse_answer(answer)
        row = {"iter": it, "time": time.strftime("%Y-%m-%d %H:%M"), "model": model, "idea": (idea or "").replace("\t", " ")}
        if code is None:
            row.update(status="no_code", seconds=round(time.time() - t0))
        else:
            cand = RUNS / f"iter_{it:04d}.py"; cand.write_text(code)
            bad = static_check(code)
            if bad:
                row.update(status=f"rejected: {bad}", seconds=round(time.time() - t0))
            else:
                res, msg, sec = evaluate(cand, a.timeout)
                if res is None:
                    row.update(status=msg[:120], seconds=round(time.time() - t0))
                else:
                    delta = round(res["pooled_mean"] - best["pooled_mean"], 5)
                    keep = delta > a.min_delta
                    row.update(status="KEPT" if keep else "discarded", pooled_mean=res["pooled_mean"], score=res["score"],
                               delta=delta, seconds=round(time.time() - t0))
                    if keep:
                        shutil.copy(cand, EXP); state["best"] = res
                        git("add", "experiment.py"); git("commit", "-m", f"iter {it}: {row['idea'][:72]} ({delta:+.5f})")
        state["iter"] = it
        state["history"].append({"status": row["status"], "score": row.get("pooled_mean", ""), "idea": row["idea"]})
        STATE.write_text(json.dumps(state, indent=1))
        log_row(row)
        print(f"[{it}] {row['status']} {row.get('pooled_mean', '')} ({row.get('delta', '')}) — {row['idea']}", flush=True)


if __name__ == "__main__":
    main()
