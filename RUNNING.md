# RUNNING.md: operating a GA run

Operational notes for starting, monitoring, and stopping an evolution run. See README.md for what the project is and what it costs.

## Prerequisites

- Python 3.11+, `pip install -r requirements.txt`
- `g++` with `-std=c++23` support on PATH (the fitness function compiles generated C++)
- A GCP service-account key at `./service-account.json` with the Vertex AI User role (see `.env.example`)
- `Dataset/APPS/` present (included in this repo)

`main.py` pre-flight checks all of these and exits with a clear error if one is missing.

## Starting a run

```bash
python3 -u main.py 2>&1 | tee run.log                 # tournament selection (default)
python3 -u main.py --scheme roulette 2>&1 | tee run.log
python3 -u main.py --scheme elitism  2>&1 | tee run.log
```

`-u` disables stdout buffering so progress prints appear live. A fresh run creates `results/run_<timestamp>_<scheme>/` and writes its config, the evolution/holdout problem split, and per-generation checkpoints there. Expect 4 to 6 hours and $5 to $6 of Vertex AI charges for a full 12-generation run.

## Monitoring

Everything is plain JSON/JSONL in the run directory:

```bash
RUN=results/run_<timestamp>_<scheme>

tail -f run.log                                  # live progress, per-call LLM log lines
cat "$RUN/costs_live.json"                       # running cost, updated after every call
jq -r '.score_reason' "$RUN/evaluations_gen00.jsonl" | sort | uniq -c   # outcome mix
jq .stats "$RUN/generation_03.json"              # fitness stats for a finished generation
```

Per-run cost reports also land in `Costs/cost_run_<name>.json` when the run finishes or is interrupted.

## Stopping and resuming

Ctrl-C (SIGINT) or SIGTERM flushes the cost report and exits. Resume later from the last completed generation checkpoint (population, history, and RNG state are restored):

```bash
python3 -u main.py --resume                      # auto-selects the newest run_*
python3 -u main.py --resume results/run_...      # or name the directory
```

## After a run

- `$RUN/summary.json`, `$RUN/best_prompt.txt`: headline result and the winning prompt.
- `run_baseline_eval.py`: evaluates evolved vs baseline prompts on the 200-problem test split into `results/comparison/` (also needs API access).
- `visualize_test_results.py`: rebuilds the five comparison figures from `results/comparison/` (offline, no API needed).

## Failure modes

| Symptom | Meaning |
|---|---|
| `service-account.json not found` | Put your GCP key file at the repo root |
| `g++ not found on PATH` in results | Install g++; every candidate scores 0 without it |
| `compile_fail` reasons dominating | Model is emitting non-code text; check `llm_errors.log` |
| 429 / RESOURCE_EXHAUSTED warnings | Vertex AI quota; the client backs off and retries up to 10 times |
| `infra_failure` entries in holdout log | All retries exhausted for that problem; scored 0 |
