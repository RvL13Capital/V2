#!/usr/bin/env python3
"""
Task Commands Generator - Generates ready-to-run commands for external terminal.

Usage:
    python scripts/task_commands.py generate-eu      # Full EU pipeline
    python scripts/task_commands.py generate-us      # Full US pipeline
    python scripts/task_commands.py train-eu         # Train EU only
    python scripts/task_commands.py detect-eu        # Detect EU patterns only
    python scripts/task_commands.py <custom_cmd>     # Wrap any command

This outputs the exact command to copy-paste into an external terminal.
"""

import sys
from pathlib import Path


COMMANDS = {
    # Full pipelines
    "generate-eu": {
        "name": "generate_eu",
        "command": "python pipeline/01_generate_sequences.py --input output/eu/detected_patterns.parquet --output-dir output/eu/sequences --skip-npy-export",
        "description": "Generate EU sequences (HDF5 only, memory-safe)"
    },
    "generate-us": {
        "name": "generate_us",
        "command": "python pipeline/01_generate_sequences.py --input output/us/detected_patterns.parquet --output-dir output/us/sequences --skip-npy-export",
        "description": "Generate US sequences (HDF5 only, memory-safe)"
    },

    # Training
    "train-eu": {
        "name": "train_eu",
        "command": "python pipeline/02_train_temporal.py --sequences output/eu/sequences/sequences.h5 --output-dir output/eu/models --epochs 100",
        "description": "Train EU model"
    },
    "train-us": {
        "name": "train_us",
        "command": "python pipeline/02_train_temporal.py --sequences output/us/sequences/sequences.h5 --output-dir output/us/models --epochs 100",
        "description": "Train US model"
    },

    # Detection
    "detect-eu": {
        "name": "detect_eu",
        "command": "python pipeline/00_detect_patterns.py --tickers EU --output-dir output/eu",
        "description": "Detect EU patterns"
    },
    "detect-us": {
        "name": "detect_us",
        "command": "python pipeline/00_detect_patterns.py --tickers US --output-dir output/us",
        "description": "Detect US patterns"
    },

    # Full pipeline with orchestrator
    "full-eu": {
        "name": "full_eu",
        "command": "python pipeline/run_robust.py --eu --both",
        "description": "Full EU pipeline (detect + train)"
    },
    "full-us": {
        "name": "full_us",
        "command": "python pipeline/run_robust.py --us --both",
        "description": "Full US pipeline (detect + train)"
    },

    # Evaluation
    "eval-eu": {
        "name": "eval_eu",
        "command": "python pipeline/evaluate_trading_performance.py --model output/eu/models/best_model.pt",
        "description": "Evaluate EU trading performance"
    },
    "eval-us": {
        "name": "eval_us",
        "command": "python pipeline/evaluate_trading_performance.py --model output/us/models/best_model.pt",
        "description": "Evaluate US trading performance"
    },
}


def print_external_command(name: str, command: str, description: str = ""):
    """Print the command ready for copy-paste."""
    print()
    print("=" * 70)
    print("COMMAND FOR EXTERNAL TERMINAL")
    print("=" * 70)
    if description:
        print(f"Task: {description}")
    print()
    print("Copy and paste this into a SEPARATE PowerShell window:")
    print()
    print("-" * 70)
    full_cmd = f'python scripts/run_external.py --name {name} "{command}"'
    print(full_cmd)
    print("-" * 70)
    print()
    print("After the task completes, tell Claude to run:")
    print(f"  python scripts/check_task.py --latest")
    print()
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/task_commands.py <command_name>")
        print()
        print("Available commands:")
        for name, info in COMMANDS.items():
            print(f"  {name:15} - {info['description']}")
        print()
        print("Or wrap any custom command:")
        print("  python scripts/task_commands.py custom \"your command here\"")
        return

    cmd_name = sys.argv[1].lower()

    # Handle custom commands
    if cmd_name == "custom" and len(sys.argv) >= 3:
        custom_cmd = sys.argv[2]
        print_external_command("custom_task", custom_cmd, "Custom command")
        return

    # Handle predefined commands
    if cmd_name in COMMANDS:
        info = COMMANDS[cmd_name]
        print_external_command(info["name"], info["command"], info["description"])
    else:
        # Treat as custom command
        custom_cmd = " ".join(sys.argv[1:])
        print_external_command("custom", custom_cmd, "Custom command")


if __name__ == "__main__":
    main()
