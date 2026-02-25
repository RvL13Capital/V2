#!/usr/bin/env python3
"""
External Task Runner - Run pipeline tasks outside Claude Code terminal.

This script runs commands in a separate process and captures all output to files,
preventing Claude Code context overflow. Claude can then read the results.

Usage (in external PowerShell/CMD window, NOT Claude Code terminal):
    python scripts/run_external.py "python pipeline/01_generate_sequences.py --input data.parquet"
    python scripts/run_external.py --name my_task "python pipeline/02_train_temporal.py"

The script will:
1. Create a timestamped task folder in output/tasks/
2. Run the command and capture stdout/stderr to files
3. Write a status.json with exit code, duration, summary
4. Claude can read status.json and tail of output via check_task.py

IMPORTANT: Run this in a separate terminal, NOT through Claude Code!
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def create_task_folder(task_name: str) -> Path:
    """Create a unique task folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{timestamp}_{task_name}"

    task_dir = Path("output/tasks") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    return task_dir


def run_command(command: str, task_dir: Path) -> dict:
    """Run command and capture output to files."""
    stdout_file = task_dir / "stdout.log"
    stderr_file = task_dir / "stderr.log"
    combined_file = task_dir / "combined.log"

    # Write command being run
    (task_dir / "command.txt").write_text(command)

    print(f"=" * 60)
    print(f"EXTERNAL TASK RUNNER")
    print(f"=" * 60)
    print(f"Task folder: {task_dir}")
    print(f"Command: {command}")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"=" * 60)
    print()
    print("Output is being captured to log files.")
    print("Progress indicator: ", end="", flush=True)

    start_time = time.time()

    # Run with output capture
    with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
         open(stderr_file, 'w', encoding='utf-8') as stderr_f, \
         open(combined_file, 'w', encoding='utf-8') as combined_f:

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        # Track progress
        line_count = 0
        last_lines = []

        # Read both streams
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)

        while True:
            for key, _ in sel.select(timeout=0.1):
                line = key.fileobj.readline()
                if line:
                    # Write to appropriate files
                    if key.fileobj is process.stdout:
                        stdout_f.write(line)
                        stdout_f.flush()
                    else:
                        stderr_f.write(line)
                        stderr_f.flush()

                    combined_f.write(line)
                    combined_f.flush()

                    # Track last lines for summary
                    last_lines.append(line.rstrip())
                    if len(last_lines) > 50:
                        last_lines.pop(0)

                    # Progress indicator
                    line_count += 1
                    if line_count % 100 == 0:
                        print(".", end="", flush=True)

            # Check if process finished
            if process.poll() is not None:
                # Read any remaining output
                remaining_stdout = process.stdout.read()
                remaining_stderr = process.stderr.read()

                if remaining_stdout:
                    stdout_f.write(remaining_stdout)
                    combined_f.write(remaining_stdout)
                    for line in remaining_stdout.split('\n'):
                        if line:
                            last_lines.append(line.rstrip())

                if remaining_stderr:
                    stderr_f.write(remaining_stderr)
                    combined_f.write(remaining_stderr)
                    for line in remaining_stderr.split('\n'):
                        if line:
                            last_lines.append(line.rstrip())

                if len(last_lines) > 50:
                    last_lines = last_lines[-50:]

                break

        sel.close()

    duration = time.time() - start_time
    exit_code = process.returncode

    print()  # New line after progress dots
    print()
    print(f"=" * 60)
    print(f"TASK COMPLETED")
    print(f"=" * 60)
    print(f"Exit code: {exit_code}")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Lines captured: {line_count}")
    print(f"=" * 60)

    # Create summary
    summary = {
        "task_id": task_dir.name,
        "command": command,
        "exit_code": exit_code,
        "success": exit_code == 0,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 1),
        "started_at": datetime.now().isoformat(),
        "lines_captured": line_count,
        "stdout_file": str(stdout_file),
        "stderr_file": str(stderr_file),
        "combined_file": str(combined_file),
        "last_50_lines": last_lines[-50:],
    }

    # Extract key metrics from output
    summary["metrics"] = extract_metrics(last_lines)

    # Write status file
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nStatus written to: {status_file}")
    print(f"\nTo check results in Claude, run:")
    print(f"  python scripts/check_task.py {task_dir.name}")

    return summary


def extract_metrics(lines: list) -> dict:
    """Extract key metrics from output lines."""
    metrics = {}

    for line in lines:
        line_lower = line.lower()

        # Look for common patterns
        if "accuracy" in line_lower or "precision" in line_lower:
            metrics["accuracy_line"] = line
        if "loss" in line_lower and ":" in line:
            metrics["loss_line"] = line
        if "epoch" in line_lower:
            metrics["last_epoch"] = line
        if "patterns" in line_lower and ("found" in line_lower or "detected" in line_lower):
            metrics["patterns_found"] = line
        if "sequences" in line_lower and ("generated" in line_lower or "created" in line_lower):
            metrics["sequences_generated"] = line
        if "error" in line_lower or "exception" in line_lower:
            if "errors" not in metrics:
                metrics["errors"] = []
            metrics["errors"].append(line)
        if "ev" in line_lower and (">" in line or "threshold" in line_lower):
            metrics["ev_metrics"] = line
        if "top-5" in line_lower or "top-15" in line_lower:
            metrics["top_k_metrics"] = line

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline tasks outside Claude Code terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_external.py "python pipeline/01_generate_sequences.py --input data.parquet"
    python scripts/run_external.py --name train_eu "python pipeline/02_train_temporal.py --epochs 100"
    python scripts/run_external.py --name detect "python pipeline/00_detect_patterns.py --tickers ALL"

IMPORTANT: Run this in a separate terminal, NOT through Claude Code!
        """
    )

    parser.add_argument(
        "command",
        help="The command to run (in quotes)"
    )
    parser.add_argument(
        "--name", "-n",
        default="task",
        help="Task name for identification (default: task)"
    )

    args = parser.parse_args()

    # Create task folder
    task_dir = create_task_folder(args.name)

    # Run the command
    try:
        summary = run_command(args.command, task_dir)
        sys.exit(summary["exit_code"])
    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
        # Write interrupted status
        status = {
            "task_id": task_dir.name,
            "command": args.command,
            "exit_code": -1,
            "success": False,
            "interrupted": True,
            "message": "Task interrupted by user (Ctrl+C)"
        }
        with open(task_dir / "status.json", 'w') as f:
            json.dump(status, f, indent=2)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running task: {e}")
        status = {
            "task_id": task_dir.name,
            "command": args.command,
            "exit_code": -1,
            "success": False,
            "error": str(e)
        }
        with open(task_dir / "status.json", 'w') as f:
            json.dump(status, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
