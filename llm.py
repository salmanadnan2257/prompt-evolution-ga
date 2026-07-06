import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Optional

import config
from cost_tracker import tracker, PRICING


class LLMCallError(RuntimeError):
    pass


_error_log_path: Path | None = None
_error_log_lock = threading.Lock()


def set_error_log_path(path: Path) -> None:
    global _error_log_path
    _error_log_path = path
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_log(line: str) -> None:
    print(line, flush=True)
    if _error_log_path is not None:
        with _error_log_lock:
            with _error_log_path.open("a") as f:
                f.write(line + "\n")


def _log_ok(msg: str) -> None:
    _write_log(f"[{time.strftime('%H:%M:%S')}] [LLM OK]   {msg}")

def _log_warn(msg: str) -> None:
    _write_log(f"[{time.strftime('%H:%M:%S')}] [LLM WARN] {msg}")

def _log_err(msg: str) -> None:
    _write_log(f"[{time.strftime('%H:%M:%S')}] [LLM ERR]  {msg}")


_CODE_GEN_DIRECTIVE = (
    "\n\n"
    "ABSOLUTE OUTPUT REQUIREMENT — NO EXCEPTIONS:\n"
    "Your entire response must consist of ONLY the raw C++ source code — one complete file, nothing else. Use C++23.\n"
    "FORBIDDEN:\n"
    "  • Backtick code fences of any form (no ```, no `, no 'cpp' or 'c++' tags)\n"
    "  • Markdown formatting of any kind\n"
    "  • Any text before the code: no explanations, no 'Here is...', no preamble\n"
    "  • Any text after the code: no summaries, no notes\n"
    "The FIRST character of your response must be the first character of the C++ source code.\n"
    "The LAST character of your response must be the last character of the C++ source code.\n"
    "Output one complete, single-file C++ program and absolutely nothing else."
)

_OPERATOR_DIRECTIVE = (
    "\n\n"
    "ABSOLUTE OUTPUT REQUIREMENT — NO EXCEPTIONS:\n"
    "Your entire response must consist of ONLY the plain text of the instruction prompt — nothing else.\n"
    "FORBIDDEN:\n"
    "  • Backtick fences or code blocks\n"
    "  • Markdown formatting\n"
    "  • Labels, headers, meta-commentary, or explanations\n"
    "The FIRST character of your response must be the first character of the prompt text.\n"
    "The LAST character must be the last character of the prompt text."
)


def _build_user_msg(question: str) -> str:
    return (
        f"{question}\n\n"
        "Write a complete, compilable C++ program that reads from stdin and writes "
        "the answer to stdout. Include main(). Compile target: g++ -O2 -std=c++23.\n\n"
        "DO NOT write any explanation, analysis, or reasoning.\n"
        "DO NOT start with 'Looking at', 'I need to', or any English text.\n"
        "Your response must begin with #include or int main, nothing before it."
    )


_api_client = None
_api_lock = threading.Lock()


def _get_client():
    global _api_client
    if _api_client is not None:
        return _api_client

    with _api_lock:
        if _api_client is not None:
            return _api_client

        try:
            from google import genai
            from google.oauth2 import service_account as _sa
        except ImportError:
            raise RuntimeError("google-genai or google-auth not installed. Run: pip install google-genai google-auth")

        sa_path = config.SERVICE_ACCOUNT_PATH
        if not sa_path.exists():
            raise RuntimeError(f"Service account file not found: {sa_path}")

        info = json.loads(sa_path.read_text())
        project = info.get("project_id", "")
        if not project:
            raise RuntimeError(f"'project_id' not found in {sa_path}.")

        creds = _sa.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        from google.genai import types as _gt
        _api_client = genai.Client(
            vertexai=True,
            project=project,
            location=config.GCP_REGION,
            credentials=creds,
            http_options=_gt.HttpOptions(timeout=120_000),
        )

        print(f"[{time.strftime('%H:%M:%S')}] [LLM]  client ready  project={project}  region={config.GCP_REGION}", flush=True)

    return _api_client


_pricing = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro":   (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    short = model.split("/")[-1]
    ri, ro = _pricing.get(short, (0.0, 0.0))
    return in_tok * ri / 1_000_000 + out_tok * ro / 1_000_000


def _call_api(phase: str, model: str, system: str, user: str,
              max_tokens: int, retries: int = 10) -> Optional[str]:
    from google.genai import types as _gt

    try:
        client = _get_client()
    except RuntimeError:
        raise

    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            is_fit = (phase == "fitness")
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=_gt.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.0 if is_fit else 0.7,
                    thinking_config=_gt.ThinkingConfig(
                        thinking_budget=512 if is_fit else 2048,
                    ),
                ),
            )
            elapsed = round(time.monotonic() - t0, 2)
            text = resp.text or ""

            # thinking tokens billed at output rate but counted separately
            usage = resp.usage_metadata
            in_tok  = getattr(usage, "prompt_token_count",     0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
            think   = getattr(usage, "thoughts_token_count",   0) or 0
            out_tok += think
            cost = _cost(model, in_tok, out_tok)

            tracker.record(phase=phase, model=model,
                           input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)

            if not text.strip():
                if attempt < retries - 1:
                    wait = 5 * (2 ** attempt) + random.uniform(0, 3)
                    _log_warn(f"empty response  phase={phase}  attempt={attempt+1}/{retries}  retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                _log_err(f"empty response  phase={phase}  GIVING UP")
                return ""

            _log_ok(
                f"phase={phase}  model={model}  attempt={attempt+1}  "
                f"elapsed={elapsed}s  in={in_tok} out={out_tok} think={think}  "
                f"cost=${cost:.5f}  running=${tracker.total():.4f}"
            )
            return text

        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            s = str(exc)
            is_rate    = "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()
            is_server  = "500" in s or "503" in s or "UNAVAILABLE" in s
            is_timeout = "timeout" in s.lower() or "timed out" in s.lower() or "timeout" in type(exc).__name__.lower()

            # print(f"api error attempt={attempt}: {s[:80]}")

            if is_timeout or is_rate or is_server:
                wait = 5 * (2 ** attempt) + random.uniform(0, 3)
                tag = "timeout" if is_timeout else ("rate_limit" if is_rate else "server_error")
                _log_warn(f"{tag}  phase={phase}  attempt={attempt+1}/{retries}  wait={wait:.0f}s")
                if attempt < retries - 1:
                    time.sleep(wait)
                continue

            _log_err(f"{type(exc).__name__}  phase={phase}  attempt={attempt+1}/{retries}  msg={s[:150]}")
            if attempt < retries - 1:
                wait = 5 * (2 ** attempt) + random.uniform(0, 3)
                time.sleep(wait)

    _log_err(f"all retries exhausted  phase={phase}  model={model}")
    return None


def _call(phase: str, model: str, system: str, user: str,
          retries: int = 10, api_max_tokens: int = 2048) -> Optional[str]:
    return _call_api(phase, model, system, user, api_max_tokens, retries)


def generate_code(system_prompt: str, question: str) -> str:
    msg = _build_user_msg(question)
    result = _call(
        phase="fitness",
        model=config.TARGET_MODEL,
        system=system_prompt + _CODE_GEN_DIRECTIVE,
        user=msg,
        api_max_tokens=config.API_FALLBACK_MAX_TOKENS_CODE,
    )
    if result is None:
        raise LLMCallError(f"generate_code: all retries failed (model={config.TARGET_MODEL})")
    return result


def run_fitness_batch(requests):
    raise NotImplementedError("Batch API not used — realtime only.")


_CROSSOVER_SYSTEM = (
    "You merge coding-instruction prompts. Output ONLY the merged prompt — "
    "no commentary, no labels, no markdown."
)

_CROSSOVER_USER = """\
Combine these two coding-instruction prompts into one coherent prompt.
Preserve the strongest strategies from each parent.
Do not simply concatenate — produce a unified instruction paragraph.

--- Parent A ---
{parent_a}

--- Parent B ---
{parent_b}
"""


def crossover(prompt_a: str, prompt_b: str) -> str:
    result = _call(
        phase="crossover",
        model=config.OPTIMIZER_MODEL,
        system=_CROSSOVER_SYSTEM + _OPERATOR_DIRECTIVE,
        user=_CROSSOVER_USER.format(parent_a=prompt_a, parent_b=prompt_b),
        api_max_tokens=config.API_FALLBACK_MAX_TOKENS_OPS,
    )
    return (result or "").strip()


_MUT_SYSTEM = (
    "You edit coding-instruction prompts. Output ONLY the modified prompt — "
    "no commentary, no labels, no markdown."
)


def mutate_inject(prompt: str, failure_examples: list | None = None) -> str:
    hint = ""
    if failure_examples:
        lines = []
        for f in failure_examples[:3]:
            line = f"  - Problem {f['problem_id']} (rating {f.get('rating','?')}): reason={f['reason']}"
            if f.get("compile_error"):
                line += f", compile_error={f['compile_error'][:80]!r}"
            elif f.get("expected"):
                line += f", expected={f['expected'][:60]!r}, actual={f['actual'][:60]!r}"
            lines.append(line)
        hint = (
            "\n\nRecent failure examples (address at least one):\n"
            + "\n".join(lines)
            + "\nAdd a strategy that targets these specific failure patterns."
        )

    user = (
        "Add one new, specific coding strategy to this instruction prompt. "
        "Examples: 'validate your output against the examples', "
        "'consider time complexity', 'check for off-by-one errors'."
        f"{hint}\n\nOutput only the modified prompt.\n\nPrompt:\n{prompt}"
    )
    result = _call(
        phase="mutate_inject",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM + _OPERATOR_DIRECTIVE,
        user=user,
        api_max_tokens=config.API_FALLBACK_MAX_TOKENS_OPS,
    )
    return (result or "").strip()


def mutate_delete(prompt: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    if len(sentences) <= 1:
        return prompt
    user = (
        "Remove one sentence from this instruction prompt — the one that "
        "contributes the least to code-generation quality. Keep everything "
        "else verbatim. Output only the modified prompt.\n\nPrompt:\n{prompt}"
    ).format(prompt=prompt)
    result = _call(
        phase="mutate_delete",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM + _OPERATOR_DIRECTIVE,
        user=user,
        api_max_tokens=config.API_FALLBACK_MAX_TOKENS_OPS,
    )
    return (result or "").strip()


def mutate_rephrase(prompt: str) -> str:
    user = (
        "Rephrase this instruction prompt using different words, but keep "
        "the same meaning and all the same strategies. "
        "Output only the rephrased prompt.\n\nPrompt:\n{prompt}"
    ).format(prompt=prompt)
    result = _call(
        phase="mutate_rephrase",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM + _OPERATOR_DIRECTIVE,
        user=user,
        api_max_tokens=config.API_FALLBACK_MAX_TOKENS_OPS,
    )
    return (result or "").strip()