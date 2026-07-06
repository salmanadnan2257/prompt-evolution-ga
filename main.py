import argparse
import json
import random
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import config
import dataset
import evaluator
import ga
import llm
import cost_tracker
from cost_tracker import tracker
from seeds import SEED_PROMPTS


def save_checkpoint(run_dir: Path, gen: int, population, stats, rng_state=None):
    gen_file = run_dir / f"generation_{gen:02d}.json"
    data = {
        "generation": gen,
        "stats":      stats,
        "rng_state":  ga.rng_state_to_dict(rng_state) if rng_state is not None else None,
        "population": [
            {
                "uid":                ind.uid,
                "prompt":             ind.prompt,
                "fitness":            ind.fitness,
                "parent_a":           ind.parent_a,
                "parent_b":           ind.parent_b,
                "operators":          ind.operators,
                "intermediate_steps": ind.intermediate_steps,
                "failure_examples":   ind.failure_examples,
            }
            for ind in population
        ],
    }
    gen_file.write_text(json.dumps(data, indent=2))


def _find_latest_run() -> Path | None:
    runs = sorted(config.RESULTS_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _load_resume_state(run_dir: Path) -> dict:
    checkpoints = sorted(run_dir.glob("generation_*.json"))
    if not checkpoints:
        raise FileNotFoundError(f"No generation_NN.json checkpoints found in {run_dir}.")

    history = []
    for ckpt in checkpoints:
        try:
            data = json.loads(ckpt.read_text())
            if "stats" in data:
                history.append(data["stats"])
        except Exception as exc:
            print(f"[RESUME] Warning: could not read {ckpt.name}: {exc}", flush=True)

    history.sort(key=lambda s: s["generation"])

    last = json.loads(checkpoints[-1].read_text())
    last_gen = last["generation"]

    population = [
        ga.Individual(
            prompt=ind["prompt"],
            fitness=ind["fitness"],
            uid=ind["uid"],
            parent_a=ind.get("parent_a", ""),
            parent_b=ind.get("parent_b", ""),
            operators=ind.get("operators", ""),
            intermediate_steps=ind.get("intermediate_steps", {}),
            failure_examples=ind.get("failure_examples", []),
        )
        for ind in last["population"]
    ]

    rng_state = last.get("rng_state")

    evo_ids_path = run_dir / "evolution_ids.json"
    evo_ids = json.loads(evo_ids_path.read_text()) if evo_ids_path.exists() else None

    rng_status = "present" if rng_state is not None else "MISSING"
    print(f"[RESUME] gen={last_gen}  population={len(population)}  history={len(history)}  rng={rng_status}", flush=True)

    return {
        "population":    population,
        "start_gen":     last_gen,
        "history":       history,
        "evolution_ids": evo_ids,
        "rng_state":     rng_state,
    }


_holdout_lock = threading.Lock()


def _holdout_one(prompt: str, prob: dict, idx: int, total: int, log_path: Path) -> dict:
    time.sleep(random.uniform(0, 2))
    print(f"[HOLDOUT] {idx+1}/{total}  id={prob['id']}  calling Gemini", flush=True)

    try:
        response = llm.generate_code(prompt, prob["question"])
    except llm.LLMCallError as exc:
        print(f"[HOLDOUT] {idx+1}/{total}  id={prob['id']}  INFRA FAILURE: {exc}", flush=True)
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
        }
        with _holdout_lock:
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

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
    }
    with _holdout_lock:
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    print(f"[HOLDOUT] {idx+1}/{total}  id={prob['id']}  score={result['score']:.2f}  reason={result['reason']}", flush=True)
    return entry


def evaluate_on_holdout(prompt: str, holdout: list[dict], run_dir: Path) -> float:
    print(f"[HOLDOUT] starting on {len(holdout)} problems", flush=True)
    log_path = run_dir / "holdout_evaluations.jsonl"
    scores = [0.0] * len(holdout)

    with ThreadPoolExecutor(max_workers=config.PARALLEL_EVALS) as executor:
        futures = {
            executor.submit(_holdout_one, prompt, prob, i, len(holdout), log_path): i
            for i, prob in enumerate(holdout)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                entry = future.result()
                scores[idx] = entry["score"] if entry["score"] >= 0.0 else 0.0
            except Exception as exc:
                print(f"[HOLDOUT] prob={idx+1} ERROR: {exc}", flush=True)
                scores[idx] = 0.0

    mean = sum(scores) / len(scores)
    print(f"[HOLDOUT] done  mean={mean:.4f}", flush=True)
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", nargs="?", const="AUTO", metavar="RUN_DIR",
        help="Resume from a previous run. Omit path to auto-select latest.",
    )
    parser.add_argument(
        "--scheme", choices=["tournament", "roulette", "elitism"], default="tournament",
        help="Selection scheme: tournament (default), roulette, elitism (tournament + elitism=1).",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}", flush=True)
    print(f"  GA PROMPT EVOLUTION — START  {ts}", flush=True)
    print(f"  Backend  : Google Cloud Vertex AI  ({config.GCP_REGION})", flush=True)
    print(f"  Target   : {config.TARGET_MODEL}", flush=True)
    print(f"  Optimizer: {config.OPTIMIZER_MODEL}", flush=True)
    print(f"  Pop={config.POPULATION_SIZE}  Gens={config.GENERATIONS}  Evo={config.EVOLUTION_SIZE}  Holdout={config.HOLDOUT_SIZE}", flush=True)
    print(f"  Scheme   : {args.scheme}", flush=True)
    print(f"{'='*65}\n", flush=True)

    print("[MAIN] pre-flight: checking Vertex AI credentials ...", flush=True)
    if not config.SERVICE_ACCOUNT_PATH.exists():
        print(f"ERROR: service-account.json not found at {config.SERVICE_ACCOUNT_PATH}")
        sys.exit(1)
    try:
        from google import genai as _check
        from google.oauth2 import service_account as _sa_check
    except ImportError as exc:
        print(f"ERROR: missing package — {exc}\nRun:  pip install google-genai google-auth", flush=True)
        sys.exit(1)

    sa = json.loads(config.SERVICE_ACCOUNT_PATH.read_text())
    print(f"[MAIN] service account OK  project={sa.get('project_id')}  email={sa.get('client_email')}", flush=True)
    del sa

    if not config.APPS_TRAIN.exists() or not config.APPS_TEST.exists():
        print("ERROR: APPS dataset directories not found.")
        print(f"  Expected train: {config.APPS_TRAIN}")
        print(f"  Expected test:  {config.APPS_TEST}")
        sys.exit(1)
    print("[MAIN] pre-flight: dataset dirs exist", flush=True)

    print("[MAIN] loading 305 pre-selected Codeforces problems ...", flush=True)
    t_load = time.time()
    all_problems = dataset.load_problems()
    print(f"[MAIN] loaded {len(all_problems)} problems in {time.time()-t_load:.1f}s", flush=True)

    if len(all_problems) < config.EVOLUTION_SIZE + config.HOLDOUT_SIZE:
        print("ERROR: not enough problems after filtering.")
        sys.exit(1)

    resume_state: dict | None = None
    run_dir: Path | None = None

    if args.resume is not None:
        if args.resume == "AUTO":
            run_dir = _find_latest_run()
            if run_dir is None:
                print("ERROR: --resume: no run_* directories found.")
                sys.exit(1)
            print(f"[RESUME] auto-selected: {run_dir}", flush=True)
        else:
            run_dir = Path(args.resume)
            if not run_dir.exists():
                print(f"ERROR: --resume: directory not found: {run_dir}")
                sys.exit(1)

        print(f"[RESUME] loading state from: {run_dir}", flush=True)
        gen_ckpts  = list(run_dir.glob("generation_*.json"))
        seed_ckpts = list(run_dir.glob("seed_??_*.json"))
        if not gen_ckpts and seed_ckpts:
            print(f"[RESUME] no gen checkpoints — found {len(seed_ckpts)} seed checkpoint(s). Resuming gen-0.", flush=True)
            resume_state = None
        else:
            try:
                resume_state = _load_resume_state(run_dir)
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}")
                sys.exit(1)

        evo_ids = resume_state["evolution_ids"] if resume_state else None
        if evo_ids is not None:
            id_set = set(evo_ids)
            evolution_pool = [p for p in all_problems if p["id"] in id_set]
            holdout        = [p for p in all_problems if p["id"] not in id_set][:config.HOLDOUT_SIZE]
            print(f"[RESUME] split restored: evo={len(evolution_pool)}  holdout={len(holdout)}", flush=True)
        else:
            evo_ids_path = run_dir / "evolution_ids.json" if run_dir else None
            if evo_ids_path and evo_ids_path.exists():
                id_set = set(json.loads(evo_ids_path.read_text()))
                evolution_pool = [p for p in all_problems if p["id"] in id_set]
                holdout        = [p for p in all_problems if p["id"] not in id_set][:config.HOLDOUT_SIZE]
                print(f"[RESUME] split from evolution_ids.json: evo={len(evolution_pool)}  holdout={len(holdout)}", flush=True)
            else:
                print("[RESUME] Warning: evolution_ids.json not found; re-splitting", flush=True)
                evolution_pool, holdout, _ = dataset.split(all_problems, config.EVOLUTION_SIZE, config.HOLDOUT_SIZE, config.RANDOM_SEED)

    else:
        print("[MAIN] splitting into evolution / holdout ...", flush=True)
        evolution_pool, holdout, rest = dataset.split(all_problems, config.EVOLUTION_SIZE, config.HOLDOUT_SIZE, config.RANDOM_SEED)
        print(f"[MAIN]   Evolution pool : {len(evolution_pool)} problems", flush=True)
        print(f"[MAIN]   Held-out test  : {len(holdout)} problems", flush=True)
        print(f"[MAIN]   Rest (unused)  : {len(rest)} problems", flush=True)

        rc = Counter(p["rating"] for p in evolution_pool)
        print("[MAIN]   Evolution pool by rating:", flush=True)
        for r in sorted(rc):
            tag = "  <- harder" if r >= 1800 else ""
            print(f"[MAIN]     {r}: {rc[r]} problems{tag}", flush=True)

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = config.RESULTS_DIR / f"run_{timestamp}_{args.scheme}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MAIN] run directory: {run_dir}", flush=True)

        (run_dir / "config.json").write_text(json.dumps({
            "population_size":  config.POPULATION_SIZE,
            "generations":      config.GENERATIONS,
            "crossover_rate":   config.CROSSOVER_RATE,
            "tournament_size":  config.TOURNAMENT_SIZE,
            "elitism_count":    config.ELITISM_COUNT,
            "mut_inject":       config.MUT_INJECT_PROB,
            "mut_delete":       config.MUT_DELETE_PROB,
            "mut_rephrase":     config.MUT_REPHRASE_PROB,
            "evolution_size":   config.EVOLUTION_SIZE,
            "holdout_size":     config.HOLDOUT_SIZE,
            "target_model":     config.TARGET_MODEL,
            "optimizer_model":  config.OPTIMIZER_MODEL,
            "gcp_region":       config.GCP_REGION,
            "random_seed":      config.RANDOM_SEED,
        }, indent=2))

        (run_dir / "evolution_ids.json").write_text(json.dumps([p["id"] for p in evolution_pool]))
    else:
        print(f"[RESUME] continuing in: {run_dir}", flush=True)

    def _on_gen(gen, pop, stats, rng_s):
        save_checkpoint(run_dir, gen, pop, stats, rng_s)
        print(f"[COST]   gen {gen:02d} done  running=${tracker.total():.4f}  best={stats['best']:.4f}  mean={stats['mean']:.4f}", flush=True)

    tracker.set_log_path(run_dir / "calls.jsonl")
    tracker.set_live_path(run_dir / "costs_live.json")
    tracker.set_costs_live_path(cost_tracker.COSTS_DIR / f"cost_{run_dir.name}.json")
    llm.set_error_log_path(run_dir / "llm_errors.log")

    if args.resume is not None:
        n = tracker.restore_from_jsonl(run_dir / "calls.jsonl")
        if n:
            print(f"[MAIN] cost restore: {n} prior calls  total=${tracker.total():.4f}", flush=True)

    print(f"[MAIN] cost log  : {run_dir}/calls.jsonl", flush=True)
    print(f"[MAIN] results   : {run_dir}\n", flush=True)

    seeds = SEED_PROMPTS[:config.POPULATION_SIZE]
    print(f"[MAIN] using {len(seeds)} seed prompts", flush=True)

    def _on_signal(signum, frame):
        print(f"\n[MAIN] signal {signum} — flushing costs and exiting ...", flush=True)
        tracker.flush()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    print("[MAIN] ════ STARTING GA ════", flush=True)
    t_ga = time.time()
    best, history = ga.run(
        seed_prompts=seeds,
        problems=evolution_pool,
        run_dir=run_dir,
        on_generation=_on_gen,
        resume_population=resume_state["population"]  if resume_state else None,
        resume_start_gen= resume_state["start_gen"]   if resume_state else 0,
        resume_history=   resume_state["history"]     if resume_state else None,
        resume_rng_state= resume_state["rng_state"]   if resume_state else None,
        selection_scheme=args.scheme,
        elitism_count=1 if args.scheme == "elitism" else None,
    )
    elapsed = time.time() - t_ga

    print(f"\n[MAIN] ════ GA DONE in {elapsed/60:.1f} min ════", flush=True)
    print(f"[MAIN] Best fitness (evolution pool): {best.fitness:.4f}", flush=True)
    print(f"[MAIN] Best prompt:\n{best.prompt}\n", flush=True)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "best_prompt.txt").write_text(best.prompt)
    print("[MAIN] history.json + best_prompt.txt saved", flush=True)

    print(f"\n[MAIN] evaluating best prompt on {len(holdout)} held-out problems ...", flush=True)
    holdout_fitness = evaluate_on_holdout(best.prompt, holdout, run_dir)
    print(f"[MAIN] Held-out fitness: {holdout_fitness:.4f}", flush=True)

    summary = {
        "elapsed_minutes":        round(elapsed / 60, 1),
        "best_prompt":            best.prompt,
        "best_fitness_evolution": best.fitness,
        "best_fitness_holdout":   holdout_fitness,
        "best_uid":               best.uid,
        "total_generations":      config.GENERATIONS,
        "resumed":                args.resume is not None,
        "selection_scheme":       args.scheme,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[MAIN] summary.json saved", flush=True)

    cost_path = tracker.flush(extra={
        "run_dir":                str(run_dir),
        "best_fitness_evolution": best.fitness,
        "best_fitness_holdout":   holdout_fitness,
    })

    print(f"\n{'='*65}", flush=True)
    print(f"  RUN COMPLETE", flush=True)
    print(f"  Evolution fitness : {best.fitness:.4f}", flush=True)
    print(f"  Holdout fitness   : {holdout_fitness:.4f}", flush=True)
    print(f"  Total cost (est.) : ${tracker.total():.4f}", flush=True)
    print(f"  Elapsed           : {elapsed/60:.1f} min", flush=True)
    print(f"  Results           : {run_dir}", flush=True)
    print(f"  Cost report       : {cost_path}", flush=True)
    print(f"{'='*65}\n", flush=True)


if __name__ == "__main__":
    main()
