#!/usr/bin/env python3
"""
Task Status Checker - For Claude to read external task results.

This script provides a context-efficient way for Claude to check on
tasks that were run externally via run_external.py.

Usage:
    python scripts/check_task.py                    # List recent tasks
    python scripts/check_task.py --latest           # Show latest task status
    python scripts/check_task.py <task_id>          # Show specific task status
    python scripts/check_task.py <task_id> --tail 100  # Show last 100 lines of output
    python scripts/check_task.py <task_id> --errors    # Show only errors

This is designed to output MINIMAL context-efficient summaries.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def get_tasks_dir() -> Path:
    """Get the tasks directory."""
    return Path("output/tasks")


def list_tasks(limit: int = 10) -> list:
    """List recent tasks."""
    tasks_dir = get_tasks_dir()

    if not tasks_dir.exists():
        return []

    tasks = []
    for task_dir in sorted(tasks_dir.iterdir(), reverse=True):
        if task_dir.is_dir():
            status_file = task_dir / "status.json"
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)
                tasks.append({
                    "id": task_dir.name,
                    "success": status.get("success", False),
                    "exit_code": status.get("exit_code", -1),
                    "duration": status.get("duration_minutes", 0),
                    "command": status.get("command", "")[:60] + "..." if len(status.get("command", "")) > 60 else status.get("command", "")
                })

        if len(tasks) >= limit:
            break

    return tasks


def get_task_status(task_id: str) -> dict:
    """Get status of a specific task."""
    tasks_dir = get_tasks_dir()

    # Find task - support partial match
    task_dir = None
    for d in tasks_dir.iterdir():
        if d.is_dir() and (d.name == task_id or task_id in d.name):
            task_dir = d
            break

    if not task_dir:
        return {"error": f"Task not found: {task_id}"}

    status_file = task_dir / "status.json"
    if not status_file.exists():
        return {"error": f"No status file for task: {task_id}"}

    with open(status_file) as f:
        return json.load(f)


def get_task_output(task_id: str, tail: int = 50, errors_only: bool = False) -> str:
    """Get output from a task."""
    tasks_dir = get_tasks_dir()

    # Find task
    task_dir = None
    for d in tasks_dir.iterdir():
        if d.is_dir() and (d.name == task_id or task_id in d.name):
            task_dir = d
            break

    if not task_dir:
        return f"Task not found: {task_id}"

    combined_file = task_dir / "combined.log"
    if not combined_file.exists():
        return "No output file found"

    with open(combined_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    if errors_only:
        error_keywords = ['error', 'exception', 'failed', 'traceback', 'fatal']
        filtered = []
        in_traceback = False
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in error_keywords):
                filtered.append(line)
                if 'traceback' in line_lower:
                    in_traceback = True
            elif in_traceback:
                filtered.append(line)
                if line.strip() and not line.startswith(' '):
                    in_traceback = False
        lines = filtered if filtered else ["No errors found in output"]

    # Return last N lines
    return "".join(lines[-tail:])


def print_compact_summary(status: dict):
    """Print a compact, context-efficient summary."""
    print("=" * 60)
    print("TASK STATUS SUMMARY")
    print("=" * 60)

    if "error" in status:
        print(f"ERROR: {status['error']}")
        return

    print(f"Task ID: {status.get('task_id', 'unknown')}")
    print(f"Success: {'YES' if status.get('success') else 'NO'}")
    print(f"Exit Code: {status.get('exit_code', -1)}")
    print(f"Duration: {status.get('duration_minutes', 0):.1f} min")
    print(f"Lines: {status.get('lines_captured', 0)}")
    print()

    # Show command (truncated)
    cmd = status.get('command', '')
    if len(cmd) > 100:
        cmd = cmd[:100] + "..."
    print(f"Command: {cmd}")
    print()

    # Show extracted metrics
    metrics = status.get('metrics', {})
    if metrics:
        print("KEY METRICS:")
        for key, value in metrics.items():
            if key == 'errors':
                print(f"  Errors: {len(value)} found")
            else:
                # Truncate long lines
                if isinstance(value, str) and len(value) > 80:
                    value = value[:80] + "..."
                print(f"  {key}: {value}")
        print()

    # Show last few lines
    last_lines = status.get('last_50_lines', [])[-10:]
    if last_lines:
        print("LAST 10 LINES:")
        for line in last_lines:
            if len(line) > 100:
                line = line[:100] + "..."
            print(f"  {line}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check status of external tasks (for Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "task_id",
        nargs="?",
        help="Task ID to check (partial match supported)"
    )
    parser.add_argument(
        "--latest", "-l",
        action="store_true",
        help="Show latest task"
    )
    parser.add_argument(
        "--list",
        type=int,
        default=0,
        metavar="N",
        help="List N recent tasks"
    )
    parser.add_argument(
        "--tail", "-t",
        type=int,
        default=0,
        metavar="N",
        help="Show last N lines of output"
    )
    parser.add_argument(
        "--errors", "-e",
        action="store_true",
        help="Show only errors from output"
    )
    parser.add_argument(
        "--full-status",
        action="store_true",
        help="Show full status JSON (verbose)"
    )

    args = parser.parse_args()

    # List tasks
    if args.list > 0:
        tasks = list_tasks(args.list)
        if not tasks:
            print("No tasks found in output/tasks/")
            return

        print("RECENT TASKS:")
        print("-" * 60)
        for t in tasks:
            status = "OK" if t['success'] else "FAIL"
            print(f"  [{status}] {t['id']}")
            print(f"       {t['command']}")
        return

    # Default: show latest if no task specified
    if not args.task_id and not args.latest:
        args.latest = True

    # Get task ID
    task_id = args.task_id
    if args.latest:
        tasks = list_tasks(1)
        if not tasks:
            print("No tasks found")
            return
        task_id = tasks[0]['id']

    # Get status
    status = get_task_status(task_id)

    if args.full_status:
        print(json.dumps(status, indent=2))
        return

    # Show summary
    print_compact_summary(status)

    # Show output if requested
    if args.tail > 0 or args.errors:
        print()
        if args.errors:
            print("ERRORS FROM OUTPUT:")
        else:
            print(f"LAST {args.tail} LINES OF OUTPUT:")
        print("-" * 60)
        output = get_task_output(task_id, tail=args.tail or 50, errors_only=args.errors)
        print(output)


if __name__ == "__main__":
    main()
