import os
import re
import signal
import subprocess
import tempfile
import time
from typing import Optional

import config


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass


_MAX_IO_LEN = 500


def extract_code(response: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+)\s*\n([\s\S]*?)```", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n?([\s\S]*?)```", response)
    if m:
        return m.group(1).strip()
    return response.strip()


def _normalise(text: str) -> list[str]:
    lines = text.rstrip("\n").split("\n")
    return [line.rstrip() for line in lines]


def _match(actual: str, expected: str) -> bool:
    return _normalise(actual) == _normalise(expected)


def _compile(code: str) -> tuple[Optional[str], str]:
    src_fd, src_path = tempfile.mkstemp(suffix=".cpp")
    bin_path = src_path[:-4]
    ok = False
    try:
        with os.fdopen(src_fd, "w") as f:
            f.write(code)

        proc = subprocess.Popen(
            ["g++", "-O2", "-std=c++23", "-o", bin_path, src_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            _, stderr = proc.communicate(timeout=config.COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return None, "compilation timed out"

        if proc.returncode != 0:
            return None, stderr[:500]

        ok = True
        return bin_path, ""

    except FileNotFoundError:
        return None, "g++ not found on PATH"
    except Exception as exc:
        return None, str(exc)
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        if not ok:
            try:
                os.unlink(bin_path)
            except OSError:
                pass


def _run_one(bin_path: str, stdin_str: str, timeout: int) -> Optional[str]:
    try:
        proc = subprocess.Popen(
            [bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            stdout, _ = proc.communicate(input=stdin_str, timeout=timeout)
            return stdout
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return None
    except Exception:
        return None


def run_tests_verbose(code: str, inputs: list, outputs: list, timeout: int = 2) -> dict:
    res: dict = {
        "score": 0.0,
        "reason": None,
        "compile": {"success": False, "error": None, "time_s": 0.0},
        "test_cases": [],
        "tests_total": len(inputs),
        "tests_run": 0,
        "early_stopped": False,
        "total_test_time_s": 0.0,
    }

    if not code or not code.strip():
        res["reason"] = "empty_code"
        return res

    t0 = time.monotonic()
    bin_path, err = _compile(code)
    res["compile"]["time_s"] = round(time.monotonic() - t0, 3)

    if bin_path is None:
        res["reason"] = "compile_fail"
        res["compile"]["error"] = err[:500] if err else "unknown"
        return res

    res["compile"]["success"] = True

    try:
        n = len(inputs)
        if n == 0:
            res["reason"] = "no_tests"
            return res

        passed = 0
        fails = 0
        t_start = time.monotonic()

        for i in range(n):
            t_tc = time.monotonic()
            actual = _run_one(bin_path, inputs[i], timeout)
            tc_time = round(time.monotonic() - t_tc, 3)

            ok = actual is not None and _match(actual, outputs[i])

            res["test_cases"].append({
                "index":    i,
                "passed":   ok,
                "actual":   (actual[:_MAX_IO_LEN] if actual is not None else None),
                "expected": outputs[i][:_MAX_IO_LEN] if isinstance(outputs[i], str) else str(outputs[i])[:_MAX_IO_LEN],
                "time_s":   tc_time,
                "timeout":  actual is None,
            })
            res["tests_run"] = i + 1

            if ok:
                passed += 1
                fails = 0
            else:
                fails += 1
                if fails >= config.EARLY_STOP_FAILURES:
                    res["early_stopped"] = True
                    break

        res["total_test_time_s"] = round(time.monotonic() - t_start, 3)
        res["score"] = passed / n

        if passed == n:
            res["reason"] = "perfect"
        elif passed == 0:
            res["reason"] = "all_fail_early_stop" if res["early_stopped"] else "all_fail"
        else:
            res["reason"] = "partial_pass"

        return res

    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def run_tests(code: str, inputs: list, outputs: list, timeout: int = 2) -> float:
    return run_tests_verbose(code, inputs, outputs, timeout)["score"]
