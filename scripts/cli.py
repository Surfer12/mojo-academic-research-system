#!/usr/bin/env python3
"""Command line interface for Mojo Academic Research Workflow System."""
import argparse
import subprocess
import os
import sys

MOJO_FILES = {
    "workflow": "academic_research_workflow.mojo",
    "match": "pattern_matcher.mojo",
    "validate": "validation_system.mojo",
    "config": "research_config.mojo",
    "example": "example_usage.mojo",
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
