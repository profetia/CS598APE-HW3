#!/usr/bin/env python3

import argparse
import csv
import datetime
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path


root = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def show_table(rows):
    table = [["commit", "min (s)", "mean (s)", "runs"]]
    for label, times in rows:
        table.append([
            label,
            "%.6f" % min(times),
            "%.6f" % (sum(times) / len(times)),
            str(len(times)),
        ])

    widths = []
    for column in range(len(table[0])):
        widths.append(max(len(row[column]) for row in table))

    for number, row in enumerate(table):
        print(" | ".join(row[n].ljust(widths[n]) for n in range(len(row))))
        if number == 0:
            print("-+-".join("-" * width for width in widths))


parser = argparse.ArgumentParser()
parser.add_argument("--commits", nargs="*")
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--size", type=int, default=1001)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--csv", action="store_true")
args = parser.parse_args()

dirty = not args.commits
commits = args.commits or ["HEAD"]
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = root / "runs" / stamp
worktree = run_dir / "worktree"
run_dir.mkdir(parents=True)

first = subprocess.check_output(
    ["git", "rev-parse", commits[0]], cwd=root, text=True
).strip()
subprocess.run(
    ["git", "worktree", "add", "--detach", str(worktree), first],
    cwd=root,
    check=True,
)

try:
    if dirty:
        patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
        if patch:
            subprocess.run(["git", "apply"], cwd=worktree, input=patch, check=True)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
        ).decode().split("\0")
        for name in untracked[:-1]:
            if name.startswith("runs/"):
                continue
            target = worktree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)

    reference = sha256(root / "identity-ref.png")
    rows = []
    raw_rows = []

    for commit in commits:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", commit], cwd=root, text=True
        ).strip()
        if not dirty:
            subprocess.run(
                ["git", "checkout", "--detach", "--force", commit_id],
                cwd=worktree,
                check=True,
                stdout=subprocess.DEVNULL,
            )

        subprocess.run(["make", "clean"], cwd=worktree, check=True)
        subprocess.run(["make", "-j4"], cwd=worktree, check=True)

        label = "working tree" if dirty else commit
        display = label + " (" + commit_id[:12] + ")"
        print("\n" + display)
        times = []
        for number in range(args.runs):
            start = time.perf_counter()
            subprocess.run(
                ["./sandpile.exe", str(args.size)],
                cwd=worktree,
                check=True,
                stdout=subprocess.DEVNULL,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(args.threads),
                },
            )
            seconds = time.perf_counter() - start
            output_hash = sha256(worktree / "identity.png")
            if args.size == 1001 and output_hash != reference:
                raise RuntimeError(
                    "incorrect output for %s: expected %s, got %s"
                    % (display, reference, output_hash)
                )
            times.append(seconds)
            raw_rows.append((display, number + 1, seconds, output_hash))
            print(
                "  %2d/%d  %.6f s  %s"
                % (number + 1, args.runs, seconds, output_hash[:12])
            )
        rows.append((display, times))

    print()
    show_table(rows)

    if args.csv:
        csv_file = run_dir / "results.csv"
        with csv_file.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["commit", "run", "seconds", "sha256"])
            writer.writerows(raw_rows)
        print("\n" + str(csv_file))
finally:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
