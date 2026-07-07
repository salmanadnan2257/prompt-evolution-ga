import json
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import dataset
import evaluator
import llm
from cost_tracker import tracker
from seeds import SEED_PROMPTS


BEST_RUN_DIR  = config.RESULTS_DIR / "run_20260429_210411"
OUT_DIR       = config.RESULTS_DIR / "comparison"

EVOLVED_JSONL  = OUT_DIR / "evolved_tournament_test_evaluations.jsonl"
BASELINE_JSONL = OUT_DIR / "baseline_test_evaluations.jsonl"
OUT_JSON       = OUT_DIR / "comparison.json"

EVOLVED_PROMPT  = (BEST_RUN_DIR / "best_prompt.txt").read_text().strip()
BASELINE_PROMPT = SEED_PROMPTS[0]

_lock = threading.Lock()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    with _lock:
        print(f"[{ts}] {msg}", flush=True)


def load_test_problems() -> list[dict]:
    log("Loading all problems ...")
    all_problems = dataset.load_problems()
    test_problems = [p for p in all_problems if p["id"].startswith("test/")]
    log(f"Loaded {len(test_problems)} test-split problems")
    return test_problems


def _eval_one(prompt: str, prob: dict, idx: int, total: int, log_path: Path) -> dict:
    time.sleep(random.uniform(0, 2))
    log(f"{log_path.stem}  {idx+1}/{total}  id={prob['id']}  calling Gemini ...")

    t0 = time.monotonic()
    try:
        response = llm.generate_code(prompt, prob["question"])
    except llm.LLMCallError as exc:
        llm_time = round(time.monotonic() - t0, 3)
        log(f"{log_path.stem}  {idx+1}/{total}  id={prob['id']}  INFRA FAILURE: {exc}")
        entry = {
            "problem_id":     prob["id"],
            "problem_rating": prob.get("rating"),
            "problem_index":  idx,
            "score":          -1.0,
            "score_reason":   "infra_failure",
            "compile":        {"success": False, "error": str(exc), "time_s": 0.0},
            "test_cases":     [],
            "tests_total":    len(prob.get("inputs", [])),
            "tests_run":      0,
            "early_stopped":  False,
            "llm_time_s":     llm_time,
        }
        with _lock:
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    llm_time = round(time.monotonic() - t0, 3)
    code = evaluator.extract_code(response)
    del response
    result = evaluator.run_tests_verbose(code, prob["inputs"], prob["outputs"], config.EXEC_TIMEOUT)
    del code

    entry = {
        "problem_id":     prob["id"],
        "problem_rating": prob.get("rating"),
        "problem_index":  idx,
        "score":          result["score"],
        "score_reason":   result["reason"],
        "compile":        result["compile"],
        "test_cases":     result["test_cases"],
        "tests_total":    result["tests_total"],
        "tests_run":      result["tests_run"],
        "early_stopped":  result["early_stopped"],
        "llm_time_s":     llm_time,
    }
    with _lock:
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    log(f"{log_path.stem}  {idx+1}/{total}  id={prob['id']}  score={result['score']:.2f}  reason={result['reason']}")
    return entry


def run_eval(label: str, prompt: str, problems: list[dict], log_path: Path) -> list[dict]:
    results: list[dict | None] = [None] * len(problems)

    done: dict[str, dict] = {}
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                e = json.loads(line)
                done[e["problem_id"]] = e
        log(f"[{label}] resuming: {len(done)} already done")

    todo = [(i, p) for i, p in enumerate(problems) if p["id"] not in done]
    log(f"[{label}] evaluating {len(todo)} remaining problems ...")

    if todo:
        with ThreadPoolExecutor(max_workers=config.PARALLEL_EVALS) as executor:
            futures = {
                executor.submit(_eval_one, prompt, prob, i, len(problems), log_path): i
                for i, prob in todo
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log(f"[{label}] ERROR idx={idx}: {exc}")
                    results[idx] = {
                        "problem_id":     problems[idx]["id"],
                        "problem_rating": problems[idx].get("rating"),
                        "problem_index":  idx,
                        "score":          -1.0,
                        "score_reason":   "unexpected_error",
                    }

    for i, prob in enumerate(problems):
        if results[i] is None:
            results[i] = done[prob["id"]]

    valid = [r["score"] for r in results if r["score"] >= 0]
    log(f"[{label}] done: mean={sum(valid)/len(valid):.4f} over {len(valid)} valid problems")
    return results


def build_comparison(evolved: list[dict], baseline: list[dict]) -> dict:
    emap = {r["problem_id"]: r for r in evolved}
    bmap = {r["problem_id"]: r for r in baseline}
    pids = [r["problem_id"] for r in evolved]

    per_problem = []
    for pid in pids:
        e = emap[pid]
        b = bmap.get(pid, {})
        per_problem.append({
            "problem_id":       pid,
            "rating":           e.get("problem_rating"),
            "evolved_score":    e.get("score", -1.0),
            "evolved_reason":   e.get("score_reason", ""),
            "evolved_compile":  e.get("compile", {}).get("success", False),
            "baseline_score":   b.get("score", -1.0),
            "baseline_reason":  b.get("score_reason", ""),
            "baseline_compile": b.get("compile", {}).get("success", False),
            "delta":            round((e.get("score") or 0) - (b.get("score") or 0), 4),
        })

    def mean_score(results):
        valid = [r["score"] for r in results if r.get("score", -1) >= 0]
        return round(sum(valid) / len(valid), 6) if valid else 0.0

    def outcome_dist(results):
        dist: dict[str, int] = defaultdict(int)
        for r in results:
            dist[r.get("score_reason", "unknown")] += 1
        return dict(dist)

    by_rating: dict[int, dict] = defaultdict(lambda: {"evolved": [], "baseline": []})
    for row in per_problem:
        r = row["rating"] or 0
        by_rating[r]["evolved"].append(row["evolved_score"])
        by_rating[r]["baseline"].append(row["baseline_score"])

    rating_breakdown = {}
    for rating in sorted(by_rating):
        es = [s for s in by_rating[rating]["evolved"]  if s >= 0]
        bs = [s for s in by_rating[rating]["baseline"] if s >= 0]
        rating_breakdown[str(rating)] = {
            "n":             len(by_rating[rating]["evolved"]),
            "evolved_mean":  round(sum(es) / len(es), 4) if es else 0.0,
            "baseline_mean": round(sum(bs) / len(bs), 4) if bs else 0.0,
        }

    e_mean = mean_score(evolved)
    b_mean = mean_score(baseline)

    return {
        "description":           "Evolved vs baseline prompt on 200 APPS test-split problems",
        "evolved_prompt":        EVOLVED_PROMPT,
        "baseline_prompt":       BASELINE_PROMPT,
        "n_problems":            len(pids),
        "evolved_mean_score":    e_mean,
        "baseline_mean_score":   b_mean,
        "delta_mean":            round(e_mean - b_mean, 6),
        "evolved_outcome_dist":  outcome_dist(evolved),
        "baseline_outcome_dist": outcome_dist(baseline),
        "by_rating":             rating_breakdown,
        "per_problem":           per_problem,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker.set_log_path(OUT_DIR / "calls.jsonl")
    llm.set_error_log_path(OUT_DIR / "llm_errors.log")

    print(f"\n{'='*60}", flush=True)
    print(f"  EVOLVED vs BASELINE: 200 APPS test-split problems", flush=True)
    print(f"  Parallel evals : {config.PARALLEL_EVALS}", flush=True)
    print(f"{'='*60}\n", flush=True)

    problems = load_test_problems()

    evolved_results  = run_eval("EVOLVED",  EVOLVED_PROMPT,  problems, EVOLVED_JSONL)
    baseline_results = run_eval("BASELINE", BASELINE_PROMPT, problems, BASELINE_JSONL)

    log("Building comparison.json ...")
    cmp = build_comparison(evolved_results, baseline_results)
    OUT_JSON.write_text(json.dumps(cmp, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"  DONE", flush=True)
    print(f"  Evolved  mean : {cmp['evolved_mean_score']:.4f}", flush=True)
    print(f"  Baseline mean : {cmp['baseline_mean_score']:.4f}", flush=True)
    print(f"  Delta         : {cmp['delta_mean']:+.4f}", flush=True)
    print(f"  Output        : {OUT_JSON}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
