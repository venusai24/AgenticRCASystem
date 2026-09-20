"""Hand-built ledgers for deterministic component tests (done-check,
verifier, critic, report): queries/coverage rows are inserted directly, so
each predicate can be driven to met / unmet / deferred without real data."""

from __future__ import annotations

import json

from agentic_rca.ledger import Ledger


class Builder:
    def __init__(self, tmp_path):
        self.rows: dict[str, list[dict]] = {}
        self.led = Ledger.create(tmp_path / "runs", result_lookup=self.rows.get)
        self.n = 0

    def q(
        self,
        rows,
        *,
        source="container_metrics",
        requested=None,
        window=(0.0, 1e10),
        tool="t",
        status="ok",
        empty_because=None,
        attempt_reason=None,
        unobserved=None,
        scope_verified=True,
        resolution=None,
        kind=None,
        error_type=None,
    ):
        self.n += 1
        short = f"q{self.n:06d}"
        qid = f"{self.led.run_id}/{short}"
        self.led.con.execute(
            "INSERT INTO queries (run_id, query_id, tool, status, error_type, row_count, empty_because, "
            "unobserved_entities, scope_verified, input_query_ids, args_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                self.led.run_id,
                qid,
                tool,
                status,
                error_type,
                len(rows),
                empty_because,
                json.dumps(unobserved) if unobserved else None,
                scope_verified,
                "[]",
                "{}",
            ],
        )
        cov_kind = kind or (
            "queried" if status == "ok" and (rows or empty_because == "no_matches") else "attempt"
        )
        self.led.con.execute(
            "INSERT INTO coverage (run_id, coverage_id, query_id, actor, kind, attempt_reason, sources, requested, "
            "requested_window, resolution, returned, unobserved_entities, scope_verified, row_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                self.led.run_id,
                f"{qid}.c1",
                qid,
                "lead",
                cov_kind,
                attempt_reason,
                json.dumps([source]),
                json.dumps(requested or {}),
                json.dumps({"start_s": window[0], "end_s": window[1]}) if window else None,
                resolution,
                "{}",
                json.dumps(unobserved) if unobserved else None,
                scope_verified,
                len(rows),
            ],
        )
        self.rows[qid] = rows
        return short

    def ev(self, qshort, rows, kind="occurrence", claim="c"):
        refs = [{"row": i, "values": {}} for i in rows]
        return self.led.apply(
            "evidence_record",
            {"claim": claim, "query_id": qshort, "row_refs": refs, "evidence_kind": kind},
            actor="lead",
        )["id"]

    def op(self, op, payload, actor="lead"):
        out = self.led.apply(op, payload, actor=actor)
        return out.get("id", out)


def link(
    cause_ev,
    effect_ev,
    *,
    ev=None,
    contrast=None,
    ct=10.0,
    et=20.0,
    ch="h1",
    eh="h1",
    crow=0,
    erow=0,
):
    evidence = [
        {"evidence_id": e, "role": "observation"} for e in (ev or sorted({cause_ev, effect_ev}))
    ]
    for c in contrast or []:
        evidence.append({"evidence_id": c, "role": "contrast"})
    return {
        "cause": "c",
        "effect": "e",
        "evidence": evidence,
        "cause_at": {"evidence_id": cause_ev, "row": crow, "ts_s": ct, "host": ch},
        "effect_at": {"evidence_id": effect_ev, "row": erow, "ts_s": et, "host": eh},
    }
