from __future__ import annotations

import json
import random
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
import evaluator
import llm


_progress_log: Path | None = None
_eval_log_dir: Path | None = None
_log_lock      = threading.Lock()
_eval_log_lock = threading.Lock()


def _log(msg: str, level: str = "INFO") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if _progress_log is not None:
            with _progress_log.open("a") as f:
                f.write(line + "\n")


def _debug(msg: str) -> None:
    _log(msg, level="DEBUG")


def _log_eval(entry: dict) -> None:
    if _eval_log_dir is None:
        return
    gen  = entry.get("generation", 0)
    path = _eval_log_dir / f"evaluations_gen{gen:02d}.jsonl"
    line = json.dumps(entry, ensure_ascii=False)
    with _eval_log_lock:
        with path.open("a") as f:
            f.write(line + "\n")


@dataclass
class Individual:
    prompt: str
    fitness: float = -1.0
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_a: str = ""
    parent_b: str = ""
    operators: str = ""
    intermediate_steps: dict = field(default_factory=dict)
    failure_examples: list = field(default_factory=list)


def _eval_code(
    gen: int,
    ind: Individual,
    prob: dict,
    idx: int,
    total: int,
    response: Optional[str],
    llm_time: float,
) -> tuple[float, Optional[dict]]:
    if response is None:
        entry = {
            "ts":                   time.strftime("%H:%M:%S"),
            "generation":           gen,
            "individual_uid":       ind.uid,
            "individual_operators": ind.operators or "seed",
            "system_prompt":        ind.prompt,
            "problem_id":           prob["id"],
            "problem_rating":       prob.get("rating"),
            "problem_index":        idx,
            "llm_call_time_s":      llm_time,
            "extracted_code":       None,
            "compile":              {"success": False, "error": "infra_failure", "time_s": 0.0},
            "test_cases":           [],
            "tests_total":          len(prob.get("inputs", [])),
            "tests_run":            0,
            "early_stopped":        False,
            "total_test_time_s":    0.0,
            "score":                -1.0,
            "score_reason":         "infra_failure",
        }
        _log_eval(entry)
        _log(f"ind={ind.uid}  prob={idx+1}/{total}  id={prob['id']}  score=-1.00  reason=infra_failure")
        return -1.0, None

    code = evaluator.extract_code(response)
    del response

    if not code or not code.strip():
        result: dict = {
            "score":            0.0,
            "reason":           "empty_response",
            "compile":          {"success": False, "error": None, "time_s": 0.0},
            "test_cases":       [],
            "tests_total":      len(prob.get("inputs", [])),
            "tests_run":        0,
            "early_stopped":    False,
            "total_test_time_s": 0.0,
        }
    else:
        result = evaluator.run_tests_verbose(
            code=code,
            inputs=prob["inputs"],
            outputs=prob["outputs"],
            timeout=config.EXEC_TIMEOUT,
        )

    score  = result["score"]
    reason = result["reason"]

    entry = {
        "ts":                   time.strftime("%H:%M:%S"),
        "generation":           gen,
        "individual_uid":       ind.uid,
        "individual_operators": ind.operators or "seed",
        "system_prompt":        ind.prompt,
        "problem_id":           prob["id"],
        "problem_rating":       prob.get("rating"),
        "problem_index":        idx,
        "llm_call_time_s":      llm_time,
        "extracted_code":       code,
        "compile":              result["compile"],
        "test_cases":           result["test_cases"],
        "tests_total":          result["tests_total"],
        "tests_run":            result["tests_run"],
        "early_stopped":        result["early_stopped"],
        "total_test_time_s":    result["total_test_time_s"],
        "score":                score,
        "score_reason":         reason,
    }
    _log_eval(entry)
    del entry, code

    _log(
        f"ind={ind.uid}  prob={idx+1}/{total}  id={prob['id']}  "
        f"rating={prob.get('rating',0)}  score={score:.2f}  reason={reason}  "
        f"llm={llm_time:.1f}s"
    )

    fail = None
    if score < 1.0 and reason != "infra_failure":
        tc_list = result.get("test_cases", [])
        first_fail = next((tc for tc in tc_list if not tc["passed"]), None)
        fail = {
            "problem_id":    prob["id"],
            "rating":        prob.get("rating"),
            "reason":        reason,
            "expected":      (first_fail["expected"][:200] if first_fail else ""),
            "actual":        (first_fail["actual"][:200] if first_fail and first_fail["actual"] else ""),
            "compile_error": (result["compile"].get("error", "")[:200] if reason == "compile_fail" else ""),
        }

    del result
    return score, fail


def _eval_one_problem(gen: int, ind: Individual, prob: dict, idx: int, total: int) -> tuple[float, Optional[dict]]:
    time.sleep(random.uniform(0, 1))

    t0 = time.monotonic()
    try:
        response = llm.generate_code(system_prompt=ind.prompt, question=prob["question"])
    except llm.LLMCallError as exc:
        llm_time = round(time.monotonic() - t0, 3)
        _log(f"ind={ind.uid}  prob={idx+1}/{total}  id={prob['id']}  INFRA FAILURE ({llm_time:.1f}s): {exc}", level="ERROR")
        return _eval_code(gen, ind, prob, idx, total, None, llm_time)

    llm_time = round(time.monotonic() - t0, 3)
    return _eval_code(gen, ind, prob, idx, total, response, llm_time)


def compute_fitness(gen: int, ind: Individual, problems: list[dict]) -> float:
    t0 = time.monotonic()
    scores = [0.0] * len(problems)
    fails: list[dict] = []
    lock = threading.Lock()

    # print(f"computing fitness for {ind.uid} gen={gen}")

    with ThreadPoolExecutor(max_workers=config.PARALLEL_EVALS) as executor:
        futures = {
            executor.submit(_eval_one_problem, gen, ind, prob, i, len(problems)): i
            for i, prob in enumerate(problems)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                score, f = future.result()
                scores[idx] = score
                if f is not None:
                    with lock:
                        fails.append(f)
            except Exception as exc:
                _log(f"ind={ind.uid}  prob={idx+1}  UNEXPECTED ERROR: {exc}", level="ERROR")
                scores[idx] = -1.0
            done += 1
            if done % 10 == 0 or done == len(problems):
                _log(f"ind={ind.uid}  progress {done}/{len(problems)}  gen={gen}")

    _pick_failures(ind, fails)
    fitness = sum(scores) / len(scores)
    elapsed = round(time.monotonic() - t0, 1)
    _log(f"ind={ind.uid}  DONE  gen={gen}  fitness={fitness:.4f}  elapsed={elapsed}s")
    return fitness


def _pick_failures(ind: Individual, fails: list[dict]) -> None:
    priority = {"compile_fail": 0, "all_fail": 1, "all_fail_early_stop": 1, "partial_pass": 2}
    fails.sort(key=lambda f: priority.get(f["reason"], 3))
    seen: set[str] = set()
    picked: list[dict] = []
    for f in fails:
        if f["reason"] not in seen or len(picked) < 3:
            picked.append(f)
            seen.add(f["reason"])
        if len(picked) >= 5:
            break
    ind.failure_examples = picked


def rng_state_to_dict(state: tuple) -> dict:
    version, internalstate, gauss_next = state
    return {"version": version, "internalstate": list(internalstate), "gauss_next": gauss_next}


def rng_state_from_dict(d: dict) -> tuple:
    return (d["version"], tuple(d["internalstate"]), d["gauss_next"])


def tournament_select(population: list[Individual], rng: random.Random) -> Individual:
    contestants = rng.sample(population, config.TOURNAMENT_SIZE)
    return max(contestants, key=lambda x: x.fitness)


def roulette_select(population: list[Individual], rng: random.Random) -> Individual:
    fits = [max(ind.fitness, 0.0) for ind in population]
    total = sum(fits)
    if total == 0:
        return rng.choice(population)
    r = rng.uniform(0, total)
    cum = 0.0
    for ind, f in zip(population, fits):
        cum += f
        if cum >= r:
            return ind
    return population[-1]


def breed(pa: Individual, pb: Individual, rng: random.Random, allow_mutation: bool = True) -> Individual:
    ops   = []
    steps: dict = {}

    combined_fails: list[dict] = []
    seen_ids: set[str] = set()
    for f in (pa.failure_examples + pb.failure_examples):
        if f["problem_id"] not in seen_ids:
            combined_fails.append(f)
            seen_ids.add(f["problem_id"])
    combined_fails = combined_fails[:5]

    if rng.random() < config.CROSSOVER_RATE:
        res = llm.crossover(pa.prompt, pb.prompt)
        prompt = res if res.strip() else rng.choice([pa.prompt, pb.prompt])
        ops.append("crossover")
    else:
        prompt = rng.choice([pa.prompt, pb.prompt])
        ops.append("copy")

    steps["after_crossover"] = prompt

    if allow_mutation:
        if rng.random() < config.MUT_INJECT_PROB:
            res = llm.mutate_inject(prompt, failure_examples=combined_fails)
            prompt = res if res.strip() else prompt
            ops.append("inject")
            steps["after_inject"] = prompt

        if rng.random() < config.MUT_DELETE_PROB:
            res = llm.mutate_delete(prompt)
            prompt = res if res.strip() else prompt
            ops.append("delete")
            steps["after_delete"] = prompt

        if rng.random() < config.MUT_REPHRASE_PROB:
            res = llm.mutate_rephrase(prompt)
            prompt = res if res.strip() else prompt
            ops.append("rephrase")
            steps["after_rephrase"] = prompt

    op_str = "+".join(ops)
    _log(f"bred  parents={pa.uid}+{pb.uid}  ops={op_str}  len={len(prompt)}")

    return Individual(
        prompt=prompt,
        parent_a=pa.uid,
        parent_b=pb.uid,
        operators=op_str,
        intermediate_steps=steps,
    )


def run(
    seed_prompts: list[str],
    problems: list[dict],
    run_dir: Path | None = None,
    on_generation=None,
    resume_population: Optional[list[Individual]] = None,
    resume_start_gen: int = 0,
    resume_history: Optional[list[dict]] = None,
    resume_rng_state: Optional[dict] = None,
    selection_scheme: str = "tournament",
    elitism_count: Optional[int] = None,
) -> tuple[Individual, list[dict]]:
    global _progress_log, _eval_log_dir
    if run_dir is not None:
        _progress_log = run_dir / "progress.log"
        _eval_log_dir = run_dir

    scheme = selection_scheme.lower()
    select = roulette_select if scheme == "roulette" else tournament_select

    elitism = elitism_count if elitism_count is not None else config.ELITISM_COUNT
    _log(f"[GA] selection={scheme}  elitism={elitism}")

    rng     = random.Random(config.RANDOM_SEED)
    history: list[dict] = list(resume_history) if resume_history else []

    if resume_population is not None:
        population = resume_population
        _log(f"[RESUME] restored {len(population)} individuals from gen {resume_start_gen}")

        if resume_rng_state is not None:
            rng.setstate(rng_state_from_dict(resume_rng_state))
            _log("[RESUME] RNG state restored")
        else:
            _log("[RESUME] WARNING: no RNG state")

        loop_start = resume_start_gen + 1

    else:
        _log(f"Evaluating {len(seed_prompts)} seeds on {len(problems)} problems ...")
        population = [Individual(prompt=p, operators="seed") for p in seed_prompts]

        existing: dict[int, dict] = {}
        if run_dir is not None:
            for ckpt in sorted(run_dir.glob("seed_??_*.json")):
                try:
                    data = json.loads(ckpt.read_text())
                    idx = data["seed_idx"]
                    existing[idx] = data
                    _log(f"[gen 0] seed checkpoint: {ckpt.name}  fitness={data['fitness']:.4f}")
                except Exception as exc:
                    _log(f"[gen 0] WARNING: could not load {ckpt.name}: {exc}", level="WARN")

        for i, ind in enumerate(population):
            if i in existing:
                saved = existing[i]
                ind.uid             = saved["uid"]
                ind.fitness         = saved["fitness"]
                ind.failure_examples = saved.get("failure_examples", [])
                _log(f"[gen 0] seed {i+1}/{len(population)}  uid={ind.uid}  RESTORED  fitness={ind.fitness:.4f}")
                continue

            _log(f"[gen 0] seed {i+1}/{len(population)}  uid={ind.uid}  evaluating ...")
            ind.fitness = compute_fitness(0, ind, problems)
            _log(f"[gen 0] seed {i+1}/{len(population)}  uid={ind.uid}  fitness={ind.fitness:.4f}")

            if run_dir is not None:
                ckpt_path = run_dir / f"seed_{i:02d}_{ind.uid}.json"
                ckpt_path.write_text(json.dumps({
                    "seed_idx":         i,
                    "uid":              ind.uid,
                    "prompt":           ind.prompt,
                    "fitness":          ind.fitness,
                    "failure_examples": ind.failure_examples,
                    "operators":        "seed",
                }))

        loop_start = 1

    stagnant = 0
    last_gen  = loop_start - 1

    for gen in range(loop_start, config.GENERATIONS + 1):
        population.sort(key=lambda x: x.fitness, reverse=True)
        fits = [x.fitness for x in population]

        valid = [f for f in fits if f >= 0.0]
        if len(valid) >= 2:
            std  = statistics.pstdev(valid)
            mean = sum(valid) / len(valid)
        else:
            std  = 0.0
            mean = sum(fits) / len(fits) if fits else 0.0

        stats = {
            "generation": gen - 1,
            "best":    max(fits),
            "mean":    sum(fits) / len(fits),
            "worst":   min(fits),
            "std":     round(std, 6),
            "cv":      round(std / mean, 6) if mean > config.CONVERGENCE_MIN_MEAN else -1.0,
            "best_uid":    population[0].uid,
            "best_prompt": population[0].prompt,
        }

        last_recorded = history[-1]["generation"] if history else -1
        if stats["generation"] > last_recorded:
            history.append(stats)

        _log(f"[gen {gen-1}] STATS  best={stats['best']:.4f}  mean={stats['mean']:.4f}  worst={stats['worst']:.4f}  std={stats['std']:.4f}")

        if on_generation:
            on_generation(gen - 1, population, stats, rng.getstate())

        cv = stats["cv"]
        if cv >= 0.0 and cv < config.CONVERGENCE_CV_THRESHOLD:
            stagnant += 1
            _log(f"[gen {gen-1}] STAGNATION {stagnant}/{config.CONVERGENCE_PATIENCE}  CV={cv:.4f}")
            if stagnant >= config.CONVERGENCE_PATIENCE:
                _log(f"[gen {gen-1}] CONVERGENCE: stopping early")
                break
        else:
            stagnant = 0

        t_gen = time.monotonic()
        _log(f"\n{'─'*60}\n  BREEDING  gen {gen}/{config.GENERATIONS}   [{time.strftime('%H:%M:%S')}]\n{'─'*60}")

        new_pop = population[:elitism]
        _log(f"[gen {gen}] elites: {[f'{e.uid} ({e.fitness:.4f})' for e in new_pop]}")

        n_off = config.POPULATION_SIZE - elitism
        _log(f"[gen {gen}] breeding {n_off} offspring ...")
        off_idx = 0
        while len(new_pop) < config.POPULATION_SIZE:
            p1 = select(population, rng)
            p2 = select(population, rng)
            allow_mut = off_idx >= config.CROSSOVER_ONLY_SLOTS
            child = breed(p1, p2, rng, allow_mutation=allow_mut)
            new_pop.append(child)
            off_idx += 1

        offspring = new_pop[elitism:]
        _log(f"[gen {gen}] evaluating {len(offspring)} offspring on {len(problems)} problems ...")

        for i, ind in enumerate(offspring):
            _log(f"[gen {gen}] offspring {i+1}/{len(offspring)}  uid={ind.uid}  ops={ind.operators}  evaluating ...")
            ind.fitness = compute_fitness(gen, ind, problems)
            _log(f"[gen {gen}] offspring {i+1}/{len(offspring)}  uid={ind.uid}  fitness={ind.fitness:.4f}")

        population = new_pop
        population.sort(key=lambda x: x.fitness, reverse=True)

        elapsed_gen = round(time.monotonic() - t_gen, 1)
        _log(f"[gen {gen}] LEADERBOARD  (took {elapsed_gen}s)")
        for rank, ind in enumerate(population, 1):
            _log(f"  #{rank}  uid={ind.uid}  fitness={ind.fitness:.4f}  ops={ind.operators or 'seed'}")

        last_gen = gen

    population.sort(key=lambda x: x.fitness, reverse=True)
    fits = [x.fitness for x in population]

    valid = [f for f in fits if f >= 0.0]
    std   = statistics.pstdev(valid) if len(valid) >= 2 else 0.0
    mean  = sum(valid) / len(valid) if valid else 0.0
    cv    = round(std / mean, 6) if mean > config.CONVERGENCE_MIN_MEAN else -1.0

    final = {
        "generation": last_gen,
        "best":   max(fits),
        "mean":   sum(fits) / len(fits),
        "worst":  min(fits),
        "std":    round(std, 6),
        "cv":     cv,
        "best_uid":    population[0].uid,
        "best_prompt": population[0].prompt,
    }
    last_recorded = history[-1]["generation"] if history else -1
    if final["generation"] > last_recorded:
        history.append(final)

    _log(f"\n{'═'*60}\n  GA DONE  gen={last_gen}  best={max(fits):.4f}  mean={sum(fits)/len(fits):.4f}\n{'═'*60}")

    return population[0], history
