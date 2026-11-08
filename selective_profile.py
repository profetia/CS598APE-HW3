#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


root = Path(__file__).resolve().parent
configs = [
    ("Off", "8195a9b"),
    ("1", "1b30743"),
    ("8", "515445f"),
    ("16", "143b1f7"),
    ("32", "041a2be"),
    ("64", "af6d752"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--size", type=int, default=1001)
args = parser.parse_args()

(root / "runs").mkdir(exist_ok=True)
worktree = root / "runs" / "selective-profile"
subprocess.run(
    ["git", "worktree", "add", "--detach", str(worktree), configs[0][1]],
    cwd=root,
    check=True,
    stdout=subprocess.DEVNULL,
)

results = []
try:
    for name, commit in configs:
        subprocess.run(
            ["git", "checkout", "--detach", "--force", commit],
            cwd=worktree,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(["make", "clean"], cwd=worktree, check=True)
        subprocess.run(["make", "-j4"], cwd=worktree, check=True)

        samples = []
        label = name if name == "Off" else name + " rows"
        print("\n" + label)
        for run in range(args.runs):
            process = subprocess.run(
                [
                    "perf",
                    "stat",
                    "-x,",
                    "-e",
                    "duration_time,cycles,instructions",
                    "./sandpile.exe",
                    str(args.size),
                ],
                cwd=worktree,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": "4"},
            )
            counters = {}
            for line in process.stderr.splitlines():
                fields = line.split(",")
                if len(fields) >= 3:
                    counters[fields[2]] = fields[0]

            seconds = int(counters["duration_time"]) / 1e9
            ipc = int(counters["instructions"]) / int(counters["cycles"])
            samples.append((seconds, ipc))
            print("  %d/%d  %.3f s  IPC %.3f" % (run + 1, args.runs, seconds, ipc))

        results.append((name, min(samples)))
finally:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=root,
        stdout=subprocess.DEVNULL,
    )

print("\nRows  Time (s)  IPC")
for name, (seconds, ipc) in results:
    print("%4s  %8.3f  %.3f" % (name, seconds, ipc))
