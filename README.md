# FusedAudit

FusedAudit is a smart-contract vulnerability auditing pipeline that fuses
three evidence layers into line-anchored findings:

    AST / structural evidence  +  LLM structured reasoning  +  Z3 / source-grounded adjudication

The pipeline slices the target contract into AST-driven evidence windows,
queries an LLM for structured vulnerability candidates, and then adjudicates
every candidate against the source with Z3 reachability constraints and dual
confidence thresholds before emitting a finding.

## Repository layout

- `run_fusedaudit.py`: command-line entry point and `audit_file` Python API.
- `fusedaudit_pipeline.py`: core audit runtime (prompt construction, provider
  calls, candidate adjudication, Z3 checking, result receipt).
- `feature_fusion.py`: source structure, AST stream, risk hints, and
  evidence-window construction.
- `candidate_decision.py`: deterministic adjudication helpers over
  source-grounded candidates.
- `source_candidates.py`: source-local candidate proposals, function-span
  parsing, and semantic admission.
- `rule_candidates.py`: opt-in rule-based candidate additions.
- `patch_context.py`: bounded paired-code context construction.
- `response_format.py`: LLM output-format protocol (JSON schema modes).
- `temporal_invariants.py`: timestamp/ordering state and path evidence.
- `fusedaudit_profiles.py`: pipeline profile and flag configuration.
- `scripts/`: raw-response contract and JSON adapters used at runtime.
- `examples/`: a small vulnerable contract for smoke testing.

## Installation

Tested with Python 3.12. Pinned versions used in our experiments:
`tree-sitter==0.25.2`, `tree-sitter-solidity==1.2.13`, `httpx==0.28.1`,
`openai==2.41.1`, `z3-solver==4.11.2.0`.

```bash
python -m pip install -r requirements_runtime.txt
```

Set your LLM endpoint (any OpenAI-compatible API):

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
```

## Quick start

```bash
python run_fusedaudit.py examples/VulnerableVault.sol --output audit.json
```

With a full audit trace:

```bash
python run_fusedaudit.py examples/VulnerableVault.sol --trace trace.json --output audit.json
```

Python API:

```python
from run_fusedaudit import audit_file

result = audit_file("examples/VulnerableVault.sol", trace_path="trace.json")
print(result["execution_receipt"])
print(result.get("findings", []))
```

Each result carries an `execution_receipt` describing the run configuration
(component switches, Z3 statistics, prompt context sizes). API credentials are
read from environment variables and are never embedded in the code.

## Datasets

The evaluation uses fault-injected Solidity contracts derived from publicly
verified sources (the SmartBugs-curated corpus, Apache-2.0) and real-world
contracts from ScaBench (MIT). Contract sources are not redistributed in this
repository; please obtain them from the upstream releases and record source
hashes together with each run for reproducibility.

## License

Released under the [MIT License](LICENSE).

## Citation

TBA.
