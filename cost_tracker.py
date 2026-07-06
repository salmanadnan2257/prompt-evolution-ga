import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PRICING = {
    "gemini-2.5-flash":  (0.15,  0.60),
    "gemini-2.5-pro":    (1.25, 10.00),
    "gemini-2.0-flash":  (0.10,  0.40),
    "gemini-2.0-flash-001": (0.10, 0.40),
    "claude-haiku-4-5@20251001":  (1.00,  5.00),
    "claude-haiku-4-5-20251001":  (1.00,  5.00),
    "claude-sonnet-4-6":          (3.00, 15.00),
}

COSTS_DIR = Path(__file__).parent / "Costs"


@dataclass
class CallRecord:
    phase: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None

    @property
    def input_cost(self) -> float:
        if self.cost_usd is not None:
            return 0.0
        rate_in, _ = PRICING.get(self.model, (0, 0))
        return self.input_tokens * rate_in / 1_000_000

    @property
    def output_cost(self) -> float:
        if self.cost_usd is not None:
            return 0.0
        _, rate_out = PRICING.get(self.model, (0, 0))
        return self.output_tokens * rate_out / 1_000_000

    @property
    def total_cost(self) -> float:
        if self.cost_usd is not None:
            return self.cost_usd
        return self.input_cost + self.output_cost


class CostTracker:

    def __init__(self):
        self.start_time: float = time.time()
        self._jsonl_path: Path | None = None
        self._live_path: Path | None = None
        self._costs_live_path: Path | None = None
        self._lock = threading.Lock()
        self._total_calls: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._phase_totals: dict = {}
        self._model_totals: dict = {}

    def set_log_path(self, path: Path) -> None:
        self._jsonl_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def set_live_path(self, path: Path) -> None:
        self._live_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def set_costs_live_path(self, path: Path) -> None:
        self._costs_live_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def restore_from_jsonl(self, path: Path) -> int:
        if not path.exists():
            return 0
        loaded = 0
        with self._lock:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    phase = entry.get("phase", "unknown")
                    model = entry.get("model", "unknown")
                    in_tok  = entry.get("in", 0)
                    out_tok = entry.get("out", 0)
                    cost    = entry.get("cost", 0.0)
                    self._total_calls += 1
                    self._total_input_tokens  += in_tok
                    self._total_output_tokens += out_tok
                    self._total_cost_usd      += cost
                    for bucket, key in [(self._phase_totals, phase), (self._model_totals, model)]:
                        if key not in bucket:
                            bucket[key] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}
                        bucket[key]["calls"] += 1
                        bucket[key]["input_tokens"]  += in_tok
                        bucket[key]["output_tokens"] += out_tok
                        bucket[key]["total_cost_usd"] = round(bucket[key]["total_cost_usd"] + cost, 6)
                    loaded += 1
            if loaded and self._live_path is not None:
                self._write_summary(self._live_path)
            if loaded and self._costs_live_path is not None:
                self._write_summary(self._costs_live_path)
        return loaded

    def record(self, phase: str, model: str,
               input_tokens: int = 0, output_tokens: int = 0,
               cost_usd: float | None = None,
               extra: dict | None = None):
        rec = CallRecord(phase=phase, model=model,
                         input_tokens=input_tokens, output_tokens=output_tokens,
                         cost_usd=cost_usd)
        call_cost = rec.total_cost
        del rec

        # print(f"record called: phase={phase} cost={call_cost:.5f}")

        with self._lock:
            self._total_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += call_cost

            for bucket, key in [(self._phase_totals, phase), (self._model_totals, model)]:
                if key not in bucket:
                    bucket[key] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}
                bucket[key]["calls"] += 1
                bucket[key]["input_tokens"] += input_tokens
                bucket[key]["output_tokens"] += output_tokens
                bucket[key]["total_cost_usd"] = round(bucket[key]["total_cost_usd"] + call_cost, 6)

            if self._jsonl_path is not None:
                entry = {
                    "t": time.strftime("%H:%M:%S"),
                    "phase": phase, "model": model,
                    "in": input_tokens, "out": output_tokens,
                    "cost": round(call_cost, 6),
                    "total_calls": self._total_calls,
                    "running_cost": round(self._total_cost_usd, 6),
                }
                if extra:
                    entry.update(extra)
                with self._jsonl_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                del entry

            if self._live_path is not None:
                self._write_summary(self._live_path)
            if self._costs_live_path is not None:
                self._write_summary(self._costs_live_path)

    def total(self) -> float:
        return self._total_cost_usd

    def _write_summary(self, path: Path, extra: dict | None = None) -> None:
        report = {
            "timestamp":                 time.strftime("%Y%m%d_%H%M%S"),
            "elapsed_seconds":           round(time.time() - self.start_time, 1),
            "mode":                      "vertex-ai-realtime",
            "total_llm_calls":           self._total_calls,
            "total_input_tokens":        self._total_input_tokens,
            "total_output_tokens":       self._total_output_tokens,
            "grand_total_estimated_usd": round(self._total_cost_usd, 6),
            "breakdown_by_phase":        self._phase_totals,
            "breakdown_by_model":        self._model_totals,
        }
        if extra:
            report["run_info"] = extra
        path.write_text(json.dumps(report, indent=2))

    def flush(self, extra: dict | None = None, path: Path | None = None) -> Path:
        if path is None:
            COSTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = COSTS_DIR / f"cost_{ts}.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._write_summary(path, extra)
        return path


tracker = CostTracker()
