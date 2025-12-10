"""
data_gen.py — tiny synthetic data generator for quick experiments.

Usage:
  python data_gen.py --out sample_logs.csv --n 50
"""

import argparse, csv, random, datetime

FAULTS = [
    ("congestion",  ["packet loss", "throughput drop", "buffer overflow", "PRB utilization 95%"]),
    ("backhaul",    ["backhaul latency", "uplink delay", "fiber flap", "L2 retransmissions"]),
    ("hardware",    ["RU failure", "PA overheating", "fan speed", "SFP error"]),
    ("interference",["RSRP low", "SINR drop", "neighbor collision", "PCI confusion"]),
    ("core",        ["UPF latency", "SMF timeouts", "AMF overload", "N3 interface errors"]),
]

def synth_line(site_id: int):
    fault, hints = random.choice(FAULTS)
    hint = random.choice(hints)
    level = random.choice(["INFO", "WARNING", "ERROR"])
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    msg = f"[{level}] gNB-{site_id}: {hint}"
    return ts, site_id, level, msg, fault

def main(out_path: str, n: int):
    random.seed(42)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","site_id","level","message","label"])
        for _ in range(n):
            ts, site, level, msg, label = synth_line(random.randint(1,64))
            w.writerow([ts, site, level, msg, label])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_logs.csv")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    main(args.out, args.n)
