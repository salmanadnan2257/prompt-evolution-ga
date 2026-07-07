# Prompt Evolution GA

A genetic algorithm that evolves system-level instruction prompts for LLM code generation. Each candidate prompt is scored by a hard, objective fitness function: Gemini generates a C++ solution for each competitive-programming problem, g++ compiles it, the binary runs against the problem's real test cases, and fitness is the fraction of tests passed. Crossover and mutation are performed by a second, stronger LLM operating directly on the prompt text.

Built for a Computational Intelligence course project (paper included in this repo, co-authored with Ghufran Alvi; the implementation here is Salman Adnan's work).

## How it works

- Chromosome: a natural-language instruction prompt (starts from 5 hand-written seeds in `seeds.py`).
- Fitness: mean pass rate of `gemini-2.5-flash`-generated C++ over 65 Codeforces-style problems from the APPS dataset, evaluated by actually compiling (`g++ -O2 -std=c++23`) and running the code against each problem's test cases with timeouts and early stopping (`evaluator.py`).
- Operators: LLM-mediated crossover (merge two parent prompts), plus three mutations, strategy injection (optionally conditioned on recent failure examples), sentence deletion, and rephrasing, all done by `gemini-2.5-pro` (`llm.py`, `ga.py`).
- Selection: tournament (default), roulette, or tournament with elitism, chosen with `--scheme`.
- Runs 12 generations with population 5, then evaluates the best prompt on a 200-problem held-out split.

Engineering around the loop: parallel fitness evaluation (`ThreadPoolExecutor`, 5 workers), per-call cost tracking with live JSON reports (`cost_tracker.py`, `Costs/`), full genealogy logging (every individual records its parents, operators, and intermediate prompt states), per-generation checkpoints with RNG state so a run can be resumed with `--resume`, and convergence-based early stopping.

## Results (from the committed artifacts in `results/`)

Three full 12-generation runs, one per selection scheme, then all three evolved prompts and the strongest hand-written seed evaluated on the same 200 held-out APPS test problems (`results/comparison/comparison.json`, regenerated figures `test_1` to `test_5`):

| Prompt | Mean pass rate (200 test problems) |
|---|---|
| Baseline (best hand-written seed) | 0.7900 |
| Tournament evolved | 0.7857 |
| Roulette evolved | 0.7717 |
| Elitism evolved | 0.7713 |

Honest summary: evolution reliably improved fitness on the 65-problem training pool (best-of-run fitness 0.65 to 0.73 depending on scheme, holdout scores 0.75 to 0.81 in each run's `summary.json`), but no evolved prompt beat the strongest hand-crafted baseline on mean test score. Tournament selection came closest and was the only scheme with a positive per-problem win/loss record against the baseline (26 wins vs 22 losses, ties elsewhere). The paper discusses why: tiny population (5), semantic drift from LLM-mediated operators, and a baseline seed that was already well tuned.

Cost: the three runs together made roughly 13,000 LLM calls and cost about $17 in Vertex AI charges ($5.99 + $6.16 + $4.90, per-run reports in `Costs/`).

All three run directories are kept in `results/` (78 MB total) because each one backs a different selection-scheme comparison in the paper and in `results/comparison/`.

## Layout

```
main.py                  entry point: pre-flight checks, GA loop, holdout eval, resume
ga.py                    GA core: Individual, selection schemes, crossover/mutation pipeline
llm.py                   Vertex AI (google-genai) calls: code generation + prompt operators
evaluator.py             compile with g++, run against test cases, score
dataset.py               load and split the APPS problem subset (Dataset/APPS)
seeds.py                 5 hand-written seed prompts
cost_tracker.py          per-call token/cost accounting, live reports into Costs/
config.py                all hyperparameters and model choices
run_baseline_eval.py     evolved-vs-baseline evaluation on the 200-problem test split
visualize_test_results.py  regenerates the 5 comparison figures from results/comparison
results/                 3 completed runs + comparison artifacts + figures
Costs/                   real cost reports from the completed runs
Evolving_LLM_Prompts...  course paper (LaTeX + PDF)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
```

Reproducing the analysis and figures needs nothing else:

```bash
python visualize_test_results.py   # rebuilds the 5 PNGs in results/comparison/
```

## Running evolution (needs GCP credentials, not verified here)

The GA itself calls Gemini models (`gemini-2.5-flash` as the code generator, `gemini-2.5-pro` as the prompt operator) through Google Cloud Vertex AI using the `google-genai` SDK. It authenticates with a GCP service-account key file at `./service-account.json` (path and region in `config.py`), no environment variables are read; see `.env.example` for the credential notes. It also requires `g++` with C++23 support on PATH and the APPS subset under `Dataset/APPS/` (included).

```bash
python main.py                     # fresh run, tournament selection
python main.py --scheme roulette   # or elitism
python main.py --resume            # continue the latest run from its last checkpoint
```

A full run makes ~4,000 to 4,700 LLM calls, takes 4 to 6 hours, and cost $5 to $6 at 2026 Vertex AI Gemini prices. Because that needs a billed GCP project, evolution was not re-run for this portfolio copy; the committed `results/` are from the original runs. What was verified here: dependencies install, all modules byte-compile, `dataset.py` loads and splits the 300 committed problems, and `visualize_test_results.py` regenerates all five figures from the committed evaluation logs. Note this machine only has `gcc-13` (no `g++`), so the compile-and-run fitness path in `evaluator.py` was reviewed but not exercised; it fails cleanly with "g++ not found on PATH" if the compiler is missing.

## Dataset and licensing

Code is MIT licensed (see LICENSE). `Dataset/APPS/` is a small subset of the APPS benchmark (Hendrycks et al., 2021), see `Dataset/APPS/README.txt` for the citation; the dataset remains under its original terms.

## Challenges

- Making fitness objective at all. The natural way to score a prompt is to ask another model whether the output "looks right," but that just moves the subjectivity around. Instead each candidate is scored by compiling the generated C++ with `g++ -O2 -std=c++23` and running it against the problem's real test cases, so fitness is the fraction of tests that actually pass (`evaluator.py`). That made the signal trustworthy but pinned the whole project to a working toolchain: no `g++` on PATH means no fitness, and the fitness path fails with a plain "g++ not found" rather than silently scoring zero.

- Exact-match output scoring is strict and sometimes wrong. Comparison is done after whitespace normalisation (`_normalise`/`_compare` in `evaluator.py`, line 41), so a correct program that prints a float with different precision, or picks a different but valid answer, is counted as a failure. That deflates every prompt's score by roughly the same amount, but it also adds noise to the fitness landscape the GA is trying to climb. Keeping the comparison exact was the honest choice given the time budget; a smarter checker per problem type would have been the right fix.

- Population of 5 is barely a population. With `POPULATION_SIZE = 5` and 12 generations (`config.py`), a single bad offspring shifts the population mean noticeably, and the scheme-to-scheme differences on the 200-problem test set are small: tournament 0.7857, roulette 0.7717, elitism 0.7713, against a baseline of 0.7900. The tournament run was the only one with a positive per-problem record versus the baseline (26 wins, 22 losses, 152 ties), and even that sits inside the noise a population that small produces. The size was a deliberate cost compromise, not a design preference.

- Cost had to be tracked or the project was uncapped. The three runs together made roughly 13,000 model calls and cost about $17 in Vertex AI charges ($5.99 + $6.16 + $4.90). Without per-call accounting it's easy to start a run, walk away, and come back to a surprise bill. `cost_tracker.py` records tokens and dollars per call and writes live JSON reports into `Costs/` during a run, so a run can be watched and killed if it drifts. The trade-off shows in the code: the tracker still carries pricing rows from an earlier backend that the project no longer uses.

- The 700-token operator cap truncates winning prompts. Crossover and mutation are done by `gemini-2.5-pro`, and their output is capped at 700 tokens (`API_FALLBACK_MAX_TOKENS_OPS` in `config.py`). The best prompt from the tournament run (`results/run_20260429_210411/best_prompt.txt`) is cut off mid-sentence at "...variable types, edge." A truncated prompt still won its run, which says the fitness signal tolerated it, but operator output should be checked for completeness before it enters the population.

- LLM-mediated operators drift. Because crossover merges two prompts and mutation rephrases them through a second model rather than swapping tokens, offspring can wander off the strategy their parents encoded (the paper calls this semantic drift, and argues selection pressure has to spend effort undoing it). This is the cost of operating on natural-language chromosomes instead of fixed-length vectors, and it interacts badly with the tiny population above.
- **Writing the exhaustive project documentation honestly.** The most significant thing this documentation pass surfaced wasn't a bug, it was a contradiction: the course paper's own abstract claims the evolved prompts outperform hand-crafted baselines, but the actual committed results show every evolved prompt scoring below the best hand-written one. Reporting that honestly, instead of smoothing it into agreement with the paper, was the real difficulty.

## What I learned

- An objective fitness function is worth more than a clever one. Grounding fitness in compile-and-run test outcomes removed a whole category of arguments about whether a prompt was "better," but it also made the project's correctness depend on a system dependency (`g++`) that has nothing to do with genetic algorithms. Deciding where fitness comes from is the most important design choice in a setup like this.

- Treating natural-language prompts as chromosomes breaks the usual GA guarantees. Standard crossover and mutation preserve structure by construction; LLM-mediated operators do not, so an offspring is only loosely related to its parents. That is why genealogy logging (every individual records its parents, operators, and intermediate prompt states) turned out to matter: without it there is no way to see whether an operator helped or just scrambled a good prompt.

- Small budgets force real engineering, not shortcuts. The population size, the 700-token operator cap, and the parallel evaluation (`ThreadPoolExecutor`, 5 workers) all exist because compute and API spend were finite. Live cost reporting and resumable per-generation checkpoints (with RNG state, so `--resume` continues deterministically) are the difference between a run you can trust and a run you have to restart from scratch when something fails four hours in.

- Negative results still need statistics. No evolved prompt beat the strongest hand-written seed on mean test score, and the gaps are fractions of a percent over 200 problems. That is a real finding, but reporting it responsibly needs a paired test or a bootstrap over problems, which the comparison script does not yet do. Getting a clean number is not the same as knowing whether it means anything.

## What I'd do differently

- Population 5 is too small. Several conclusions in the paper (scheme comparisons especially) sit inside the noise a population that small produces, and a single bad offspring moves the population mean a lot. More, cheaper generations with a bigger population would have been a better spend of the same $17.
- Cap the operators less aggressively. The tournament run's winning prompt in `results/run_20260429_210411/best_prompt.txt` is visibly cut off mid-sentence ("...variable types, edge"), an artifact of the 700-token output cap on crossover/mutation calls. A truncated prompt winning the run says the fitness signal tolerated it, but it's still a bug: operator outputs should be validated for completeness before entering the population.
- No statistical testing. Evolved vs baseline differs by half a point of pass rate on 200 problems; a paired test (or bootstrap over problems) should have been in the comparison script before claiming anything in either direction.
- Decouple config from code. Model names, region, and the service-account path are constants in `config.py`; environment variables or a CLI flag would make runs reproducible without editing source. Same for the hardcoded `g++` binary name in `evaluator.py`.
- The stdout comparison in `evaluator.py` is exact-match after whitespace normalisation. APPS problems with floating-point answers or multiple valid outputs get scored as failures, which deflates every prompt's score and adds noise to fitness.
- Minor hygiene: `main.py` prints "loading 305 pre-selected problems" but the committed dataset has 300, and `cost_tracker.py` still carries Claude pricing rows from an earlier iteration of the project that used a different backend.
