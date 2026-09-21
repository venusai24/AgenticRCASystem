# Agentic RCA System

An agentic root-cause-analysis (RCA) system that reasons over incident telemetry (traces, metrics, logs) to produce evidence-grounded, verifiable causal hypotheses. Unlike systems that merely summarize anomalies, this scaffold explicitly constructs an investigative loop to pinpoint root causes without relying on pre-encoded domain knowledge.

## Features

- **Hypothesis-Driven Investigation (Abductive Loop)**
  Instead of blindly searching through logs, the system formulates testable predictions and uses them to direct the investigation. It keeps at least two hypotheses alive until one is definitively refuted by evidence.
- **External Structured Ledger**
  Prevents unbounded context growth. The system maintains a persistent ledger of gathered evidence, failed queries, and evolving hypotheses outside the LLM context.
- **Context-Isolated Adversarial Critic**
  A separate agent in a fresh context reviews the investigation's conclusions, constructing strong alternative explanations and checking for logical gaps or unverified claims.
- **Deterministic Citation Verifier**
  To prevent LLM hallucination and sycophancy, a non-LLM layer re-executes all cited queries to ensure the evidence exists exactly as claimed.
- **Lossless Ingestion**
  Ingests `traces.csv`, `metrics.csv`, and `logs.csv` into a fast, columnar DuckDB store with time, service, and trace indexing. It handles unaligned clocks by dynamically estimating cross-host clock offsets from parent/child span timings.
- **Immutable Run Bundles**
  Each investigation is frozen into a self-contained bundle including the full ledger, every query executed, the critic's objections, and data-sufficiency declarations.
- **Human Collaboration**
  You can inject human hypotheses ("hunches") into the investigation, and the agent will subject them to the exact same rigorous evidence gathering as its own theories.

---

## Installation & Setup

1. **Clone and Install**
   Ensure you have Python 3 installed. Install the package using the provided `pyproject.toml`:
   ```bash
   pip install -e .
   ```
   Or explicitly set `PYTHONPATH` when running scripts:
   ```bash
   export PYTHONPATH=$(pwd)/src
   ```

2. **Environment Variables**
   Ensure your LLM provider API keys are set. E.g.:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
   *(If an `.env` file is present in the root directory, it will be automatically picked up.)*

---

## Usage

The Agentic RCA System exposes a command-line interface via `python -m agentic_rca`.

### 1. Ingest Data

Before investigating, you must ingest the CSV incident data (`traces.csv`, `metrics.csv`, `logs.csv`) into a structured DuckDB database.

```bash
python -m agentic_rca ingest \
    --data incident_data/ \
    --store run.duckdb
```

### 2. Run an Investigation

Initiate an agentic investigation. You must provide the incident timestamp (the time it was detected, not necessarily when it started).

```bash
python -m agentic_rca run \
    --store run.duckdb \
    --incident-ts "2021-03-04T10:00:00Z" \
    --max-steps 30
```

**Providing Human Hypotheses:**
You can seed the agent with your own hunches:
```bash
python -m agentic_rca run \
    --store run.duckdb \
    --incident-ts "2021-03-04T10:00:00Z" \
    --hypothesis "I think the connection pool is exhausted on the API gateway" \
    --hypothesis "Memory leak in Service A"
```

### 3. Ask Questions Over a Run Bundle

Once an investigation is concluded and a run bundle is frozen, you can ask questions against the resulting bundle and the ledger.

```bash
python -m agentic_rca answer \
    --bundle path/to/bundle.json \
    --question "Did we check the database latency during the spike?"
```

### 4. Verify a Run Bundle

Recomputes the cryptographic hashes and verifies the parent chain integrity of an investigation bundle to ensure it hasn't been tampered with.

```bash
python -m agentic_rca verify-bundle \
    --bundle path/to/bundle.json
```

---

## Architecture Overview

1. **Stage 0 (Ingest & Index)**: Deterministic loading into DuckDB. Extracts log templates and derives call topology without assuming causality.
2. **Stage 1 (Orientation)**: The Lead Agent reviews a dataset manifest and states all potential conditions that could produce the symptoms *before* querying.
3. **Stage 2 (Investigation Loop)**: The core loop. The Lead queries data, updates the ledger, refutes/proposes hypotheses, and handles tool failures. 
4. **Stage 3 (Adversarial Challenge)**: A Critic reviews the proposed conclusion from a fresh context. 
5. **Stage 4 (Verification)**: Non-LLM deterministic re-execution of queries. Small-context entailment checks on the citations.
6. **Stage 5 (Report & Freeze)**: Compiles the ledger into a reproducible run bundle.
