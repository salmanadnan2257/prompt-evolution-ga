import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import config

_SELECTED: dict[str, dict[int, list[str]]] = {
    "train": {
        800:  ["0038","0044","0066","0076","0081","0082","0105","0113","2094",
               "2179","2197","2207","2346"],
        1000: ["0048","0054","0071","0092","0095","0121","1606"],
        1100: ["0012","0094","0103","2160","2203","2334","2366","2381"],
        1200: ["0036","0045","0091","0098","2029","2040","2099","2227","2371",
               "2378","2392"],
        1300: ["0004","0010","0024","0107","0108","2056","2115","2151","2195",
               "2211","2320"],
        1400: ["0005","0008","0032","0046","0063","0116","2017","2057","2064",
               "2071","2080","2188"],
        1500: ["0028","0065","0085","2043","2065","2087","2142","2216","2248",
               "2285","2335","2337","2347","2362"],
        1600: ["0001","0062","0067","1605","2002","2010","2023","2069","2089"],
        1700: ["0042","0050","0120","2001","2008","2039","2046","2083","2138",
               "2168"],
        # harder problems always in the evo pool
        1800: ["2015","2018","2028"],
        1900: ["2005","2007"],
    },
    "test": {
        800:  ["0017","0021","0050","0067","0083","0087","0124","0131","0145",
               "0157","0163","0168","0171","0184","0202","0216","0225","0233",
               "0271","0276","0282"],
        1000: ["0020","0076","0095","0102","0107","0146","0172","0177","0185",
               "0187","0195","0198","0204","0207","0215","0224","0244","0259",
               "0273","0310","0315","0333","0338","0347","0354","0390","0395",
               "0397","0398","0405","0412","0419","0426","0433","0437","0438",
               "0443","0446","0448","0481"],
        1100: ["0015","0016","0037","0046","0057","0061","0065","0070","0101",
               "0113","0116","0121","0158","0175","0189","0231","0234","0263",
               "0269"],
        1200: ["0002","0003","0004","0006","0007","0008","0009","0010","0011",
               "0012","0013","0014","0018","0019","0022","0023","0024","0026",
               "0028","0029","0030","0031","0033","0034","0035","0036","0039",
               "0040","0041","0042","0043","0044","0045","0047","0048","0049",
               "0051","0052","0053","0054","0055","0056","0058","0060","0062",
               "0063","0064","0066","0068","0069","0071","0072","0073","0074",
               "0078","0079","0080","0081","0082","0084","0085","0086","0090",
               "0091","0093","0094","0096","0097","0098","0099","0100","0105",
               "0106","0109","0110","0111","0112","0114","0115","0118","0119",
               "0120","0122","0123","0125","0126","0127","0128","0129","0130",
               "0132","0133","0134","0135","0136","0137","0138","0139","0140",
               "0141","0142","0143","0144","0147"],
        1300: ["0000","0001","0005","0032","0038","0088","0089","0092","0103",
               "0104","0108"],
        1400: ["0025","0027","0059","0075","0077"],
    },
}

# how many per tier for the 60-problem regular evolution pool (800-1700)
_EVOLUTION_PER_TIER: dict[int, int] = {
    800:  7,
    1000: 6,
    1100: 6,
    1200: 7,
    1300: 7,
    1400: 7,
    1500: 7,
    1600: 6,
    1700: 7,
}

_RESERVED_PER_TIER: dict[int, int] = {
    800:  6,
    1000: 1,
    1100: 2,
    1200: 4,
    1300: 4,
    1400: 5,
    1500: 7,
    1600: 3,
    1700: 3,
}


def load_problems() -> list[dict]:
    sys.set_int_max_str_digits(1_000_000)
    problems: list[dict] = []

    dirs = {"train": config.APPS_TRAIN, "test": config.APPS_TEST}

    for split, ratings in _SELECTED.items():
        base = dirs[split]
        for rating, folders in ratings.items():
            for folder in folders:
                prob_dir = base / folder
                q_f  = prob_dir / "question.txt"
                io_f = prob_dir / "input_output.json"

                if not (q_f.exists() and io_f.exists()):
                    print(f"  [warn] missing files for {split}/{folder}, skipping")
                    continue

                try:
                    io = json.loads(io_f.read_text())
                except Exception:
                    print(f"  [warn] bad JSON for {split}/{folder}, skipping")
                    continue

                inputs  = io.get("inputs", [])
                outputs = io.get("outputs", [])

                if not inputs or not outputs:
                    print(f"  [warn] empty test cases for {split}/{folder}, skipping")
                    continue

                problems.append({
                    "id":       f"{split}/{folder}",
                    "question": q_f.read_text().strip(),
                    "inputs":   inputs,
                    "outputs":  outputs,
                    "rating":   rating,
                })

    return problems


def split(
    problems: list[dict],
    evolution_size: int,
    holdout_size: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)

    # harder train problems (rating >= 1800) always go into evo pool
    harder = [p for p in problems if p["id"].startswith("train/") and p["rating"] >= 1800]

    by_tier: dict[int, list[dict]] = defaultdict(list)
    for p in problems:
        if p["id"].startswith("train/") and p["rating"] < 1800:
            by_tier[p["rating"]].append(p)

    evo_regular: list[dict] = []
    for rating in sorted(by_tier):
        tier = by_tier[rating].copy()
        rng.shuffle(tier)
        n = _EVOLUTION_PER_TIER.get(rating, 0)
        evo_regular.extend(tier[:n])

    evolution = harder + evo_regular

    evo_ids = {p["id"] for p in evolution}
    rest = [p for p in problems if p["id"] not in evo_ids]
    rng.shuffle(rest)

    holdout = rest[:holdout_size]
    leftover = rest[holdout_size:]

    return evolution, holdout, leftover


def reserved_train_problems(problems: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    by_tier: dict[int, list[dict]] = defaultdict(list)
    for p in problems:
        if p["id"].startswith("train/") and p["rating"] < 1800:
            by_tier[p["rating"]].append(p)

    reserved: list[dict] = []
    for rating in sorted(by_tier):
        tier = by_tier[rating].copy()
        rng.shuffle(tier)
        n_evo = _EVOLUTION_PER_TIER.get(rating, 0)
        n_res = _RESERVED_PER_TIER.get(rating, 0)
        reserved.extend(tier[n_evo: n_evo + n_res])

    return reserved
