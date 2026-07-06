from pathlib import Path

ROOT         = Path(__file__).parent
APPS_TRAIN   = ROOT / "Dataset" / "APPS" / "train"
APPS_TEST    = ROOT / "Dataset" / "APPS" / "test"
RESULTS_DIR  = Path(__file__).parent / "results"

# 65 problems for evolution, 200 held out for test
EVOLUTION_SIZE  = 65
HOLDOUT_SIZE    = 200
RANDOM_SEED     = 42
EARLY_STOP_FAILURES = 3

POPULATION_SIZE   = 5
GENERATIONS       = 12
ELITISM_COUNT        = 0
CROSSOVER_ONLY_SLOTS = 2
TOURNAMENT_SIZE      = 3

CROSSOVER_RATE      = 0.8

MUT_INJECT_PROB   = 0.25
MUT_DELETE_PROB   = 0.15
MUT_REPHRASE_PROB = 0.10

# vertex ai / gemini setup
SERVICE_ACCOUNT_PATH = Path(__file__).parent / "service-account.json"
GCP_REGION           = "us-central1"

TARGET_MODEL    = "gemini-2.5-flash"
OPTIMIZER_MODEL = "gemini-2.5-pro"

TARGET_TEMPERATURE    = 0.0
OPTIMIZER_TEMPERATURE = 0.7

API_FALLBACK_MAX_TOKENS_CODE = 8192
API_FALLBACK_MAX_TOKENS_OPS  = 700

# stop early if population is homogeneous for PATIENCE consecutive gens
CONVERGENCE_CV_THRESHOLD = 0.01
CONVERGENCE_PATIENCE     = 3
CONVERGENCE_MIN_MEAN     = 0.01

EXEC_TIMEOUT    = 2
COMPILE_TIMEOUT = 15
PARALLEL_EVALS  = 5
