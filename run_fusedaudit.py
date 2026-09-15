"""Command-line entry point for the FusedAudit pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusedaudit_pipeline import run_audit


def audit_file(source_path: str | Path, trace_path: str | Path | None = None) -> dict:
    """Audit one Solidity source file and verify the released configuration."""

    result = run_audit(str(source_path), trace_path=str(trace_path) if trace_path else None)
    receipt = result.get("execution_receipt") or {}
    if receipt.get("retrieval_enabled") is not False:
        raise RuntimeError("configuration check failed: retrieval flag mismatch")
    if receipt.get("retrieval_disabled") is not True:
        raise RuntimeError("configuration check failed: receipt incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FusedAudit audit pipeline on a Solidity source file."
    )
    parser.add_argument("source", type=Path, help="Path to a Solidity source file")
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional path for the audit trace JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the final JSON result",
    )
    args = parser.parse_args()

    result = audit_file(args.source, args.trace)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
