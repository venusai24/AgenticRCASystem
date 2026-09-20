"""Stage 4, layer 1: the deterministic citation verifier (§4.1).

Re-executes the query behind every piece of evidence in scope
(design-d6-d7 §11: the chain, factors, refutation bases, symptoms,
ordering notes and objection resolutions) through the registry's
`reexecute` -- the same code path, never replayed SQL -- and checks that
the cited rows and values are really there, and that each link's anchor
timestamp and host are that row's. No LLM is involved, so this layer
cannot be sycophantic; a test asserts no LLM module is imported.
"""

from __future__ import annotations

import json

from agentic_rca.donecheck import verifier_scope
from agentic_rca.ledger import Ledger, values_match
from agentic_rca.tools._common import ENTITY_KINDS, TIME_COLUMNS
from agentic_rca.tools.registry import ToolRuntime


class VerifierRefused(Exception):
    """The data under the store is not the data the run used."""


def _row_ts(row: dict) -> float | None:
    for c in TIME_COLUMNS:
        if isinstance(row.get(c), (int, float)):
            return row[c] / 1000.0 if c == "timestamp_ms" else float(row[c])
    return None


def _row_host(row: dict) -> str | None:
    for c in ENTITY_KINDS["host"]:
        if row.get(c) is not None:
            return str(row[c])
    return None


def _find(rows: list[dict], ref: dict, order_total: bool) -> dict | None:
    """Values are authoritative; the ordinal is checked only when the order
    is total (contracts-b1 §8)."""

    def matches(r):
        return all(c in r and values_match(r[c], v) for c, v in ref["values"].items())

    if order_total:
        if ref["row"] < len(rows) and matches(rows[ref["row"]]):
            return rows[ref["row"]]
        return None
    return next((r for r in rows if matches(r)), None)


def verify_evidence(ledger: Ledger, runtime: ToolRuntime, eid: str) -> dict:
    st = ledger.state()
    e = st.evidence[eid]
    qrow = ledger.query_row(e["query_id"])
    base = {"evidence_id": eid, "query_id": e["query_id"]}
    if qrow is None or qrow["status"] != "ok":
        return dict(
            base, verdict="unverifiable", detail="the cited query is not a successful logged query"
        )
    try:
        res = runtime.reexecute(qrow)
    except Exception as exc:  # noqa: BLE001 - any re-execution failure is 'unverifiable', recorded
        return dict(
            base, verdict="unverifiable", detail=f"re-execution failed: {type(exc).__name__}: {exc}"
        )
    rows, total = res["rows"], res["order_total"]
    digest_match = res["digest"] == qrow["result_digest"]
    base["digest_match"] = digest_match
    if not e["row_refs"]:
        if rows:
            return dict(
                base,
                verdict="fail",
                detail=f"cited as an empty result, but re-execution returned {len(rows)} rows",
            )
        return dict(base, verdict="pass", detail="empty result reproduced")
    found: dict[int, dict] = {}
    for ref in e["row_refs"]:
        r = _find(rows, ref, total)
        if r is None:
            return dict(
                base,
                verdict="fail",
                detail=f"cited row {ref['row']} with {ref['values']} not found in the re-executed result",
            )
        found[ref["row"]] = r
    # anchors: ts_s and host must be the cited row's own
    for lid, lk in st.links.items():
        if lk.get("retired"):
            continue
        for end in ("cause_at", "effect_at"):
            a = lk.get(end)
            if not a or a["evidence_id"] != eid:
                continue
            r = found.get(a["row"])
            if r is None:
                return dict(
                    base,
                    verdict="fail",
                    detail=f"{ledger.short(lid)}.{end} cites row {a['row']}, which the evidence doesn't cite",
                )
            ts, host = _row_ts(r), _row_host(r)
            if ts is None or not values_match(ts, a["ts_s"]):
                return dict(
                    base,
                    verdict="fail",
                    detail=f"{ledger.short(lid)}.{end}.ts_s={a['ts_s']} but the row's timestamp is {ts}",
                )
            if (a.get("host") or None) != host:
                return dict(
                    base,
                    verdict="fail",
                    detail=f"{ledger.short(lid)}.{end}.host={a.get('host')!r} but the row's host is {host!r}",
                )
    return dict(
        base,
        verdict="pass",
        detail="cited values reproduced"
        + ("" if digest_match else " (result digest drifted; values match)"),
    )


def verify(
    ledger: Ledger,
    runtime: ToolRuntime,
    *,
    current_dataset_digest: dict | None = None,
    evidence_ids: list[str] | None = None,
) -> dict:
    """Verify every evidence in scope and write one `evidence_verdict`
    event each. Returns counts and the per-evidence verdicts."""
    run_digest = json.loads(ledger.run_row().get("dataset_digest") or "{}")
    if current_dataset_digest is not None and run_digest and run_digest != current_dataset_digest:
        raise VerifierRefused(
            "the dataset digest differs from the one this run recorded; re-execution would verify against different data"
        )
    st = ledger.state()
    scope = evidence_ids if evidence_ids is not None else verifier_scope(st, st.latest_conclusion())
    verdicts = []
    for eid in scope:
        v = verify_evidence(ledger, runtime, eid)
        ledger.apply("evidence_verdict", v, actor="scaffold")
        verdicts.append(v)
    counts = {
        k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("pass", "fail", "unverifiable")
    }
    return {"counts": counts, "verdicts": verdicts}
