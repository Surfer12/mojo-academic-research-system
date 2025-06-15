#!/usr/bin/env python3
"""Command line interface for Mojo Academic Research Workflow System."""
import argparse
import subprocess
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (mojo-academic-research-system)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MOJO_FILES = {
    "workflow": os.path.join(PROJECT_ROOT, "academic_research_workflow.mojo"),
    "match": os.path.join(PROJECT_ROOT, "pattern_matcher.mojo"),
    "validate": os.path.join(PROJECT_ROOT, "validation_system.mojo"),
    "config": os.path.join(PROJECT_ROOT, "research_config.mojo"),
    "example": os.path.join(PROJECT_ROOT, "example_usage.mojo"),
}

MOJO_BIN = os.environ.get("MOJO_BIN", "mojo")


def run_module(name: str, mojo_args: list[str]):
    """Run the specified Mojo module with optional arguments."""
    if name not in MOJO_FILES:
        raise ValueError(f"Unknown command: {name}")

    path = MOJO_FILES[name]
    cmd = [MOJO_BIN, "run", path]
    if mojo_args:
        cmd.append("--")
        cmd.extend(mojo_args)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run Mojo research workflow modules from the command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd_name, file in MOJO_FILES.items():
        sub = subparsers.add_parser(cmd_name, help=f"Run {file}")
        sub.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the Mojo program",
        )

    args = parser.parse_args()
    run_module(args.command, args.args)


if __name__ == "__main__":
    main()
