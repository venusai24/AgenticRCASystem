"""The done-check (design §4.4, design-d6-d7 Part I): one pure,
deterministic function over the folded ledger, coverage rows, query rows
and a clock-offset snapshot. No I/O, no LLM import (asserted by a test).

`precheck_for(...)` is the thin adapter the loop and orchestrator use: it
assembles the input from a run, calls `done_check`, records the result as a
`done_check_result` scaffold event and renders the unmet items for the lead
from fixed, domain-free templates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

from agentic_rca.ledger.fold import LedgerState, two_live_check

GATING_PRE = {1, 2, 3, 4, 5, 8, 9, 10, 11}


@dataclass
class DoneCheckInput:
    state: LedgerState
    coverage: list[dict]  # coverage rows, current run + lineage (JSON columns decoded)
    queries: dict[str, dict]  # query_id -> {tool, input_query_ids(list), status}
    clock_offsets: list[dict]  # run snapshot: host_a, host_b, offset_ms, residual_std_ms
    source_ts_unit_s: dict[str, float]
    phase: str = "pre_challenge"  # pre_challenge | final
    critic_rounds_done: int = 0
    residual_k: float = 2.0


@dataclass
class DoneCheckResult:
    phase: str
    kind: str | None
    conclusion_seq: int | None
    ok: bool
    criteria: dict[str, dict]
    unmet: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)
    observability_gaps: dict = field(default_factory=lambda: {"on_path": [], "other": []})
    candidates: list[dict] | None = None
    inputs: dict = field(default_factory=dict)


def _item(c: int, code: str, target: str | None = None, **detail) -> dict:
    return {"criterion": c, "code": code, "target": target, "detail": detail}


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class _Ctx:
    def __init__(self, inp: DoneCheckInput):
        self.inp = inp
        self.st = inp.state
        self.cov_by_q: dict[str, list[dict]] = {}
        for c in inp.coverage:
            self.cov_by_q.setdefault(c["query_id"], []).append(c)
        self.flags: list[dict] = []
        self.stripped: list[str] = []
        self.pairs = {}
        for r in inp.clock_offsets:
            a, b = sorted([r["host_a"], r["host_b"]])
            self.pairs[(a, b)] = r

    # ---- filter F (§2) --------------------------------------------------------------
    def survives(self, eid: str) -> bool:
        if eid not in self.st.evidence:
            return False
        if self.inp.phase != "final":
            return True
        v = self.st.verdicts.get(eid)
        if v is None or v.get("verdict") != "pass":
            if v is not None and v.get("verdict") == "fail" and eid not in self.stripped:
                self.stripped.append(eid)
            return False
        return True

    def guard_ok(self, eid: str) -> bool:
        return self.st.evidence[eid].get("guard", "ok") == "ok"

    def contrast_ok(self, ev_role: dict) -> bool:
        eid = ev_role["evidence_id"]
        return ev_role.get("role") == "contrast" and self.survives(eid) and self.guard_ok(eid)

    # ---- query scope S(q) ------------------------------------------------------------
    def scope(self, qid: str, _depth: int = 0) -> dict:
        out = {"sources": set(), "entities": {}, "windows": [], "resolution": None}
        q = self.inp.queries.get(qid, {})
        for c in self.cov_by_q.get(qid, []):
            out["sources"] |= set(c["sources"] or [])
            for d in (c.get("requested") or {}, c.get("returned") or {}):
                for kind, vals in d.items():
                    if isinstance(vals, list):
                        out["entities"].setdefault(kind, set()).update(map(str, vals))
            out["windows"].append(c.get("requested_window"))
            if c.get("resolution") and not _isnan(c["resolution"]):
                out["resolution"] = c["resolution"]
        if _depth < 8:
            for iq in q.get("input_query_ids") or []:
                sub = self.scope(iq, _depth + 1)
                out["sources"] |= sub["sources"]
                for k, v in sub["entities"].items():
                    out["entities"].setdefault(k, set()).update(v)
                out["windows"] += sub["windows"]
        return out

    def gran(self, eid: str) -> float:
        qid = self.st.evidence[eid]["query_id"]
        s = self.scope(qid)
        if s["resolution"]:
            return float(s["resolution"])
        units = [
            self.inp.source_ts_unit_s[t] for t in s["sources"] if t in self.inp.source_ts_unit_s
        ]
        return max(units) if units else 1.0

    def single_source(self, eid: str) -> str | None:
        srcs = self.scope(self.st.evidence[eid]["query_id"])["sources"]
        return next(iter(srcs)) if len(srcs) == 1 else None


# ---- criteria ----------------------------------------------------------------------------


def _links_of(st: LedgerState, h: dict) -> list[str]:
    return [lid for b in h["branches"] for lid in b["links"]]


def _terminal(h: dict) -> str | None:
    for b in h["branches"]:
        if b["joins"] is None and b["links"]:
            return b["links"][-1]
    return None


def _c1(x: _Ctx, h: dict, concl: dict) -> list[dict]:
    items = []
    links = _links_of(x.st, h)
    if not links:
        return [_item(1, "no_links", h["id"])]
    for lid in links:
        lk = x.st.links[lid]
        if not any(x.survives(e["evidence_id"]) for e in lk["evidence"]):
            items.append(_item(1, "link_without_evidence", lid))
        for end in ("cause_at", "effect_at"):
            a = lk.get(end)
            if a and not x.survives(a["evidence_id"]):
                items.append(
                    _item(1, "anchor_evidence_excluded", lid, end=end, evidence=a["evidence_id"])
                )
    if sum(1 for b in h["branches"] if b["joins"] is None) != 1:
        items.append(_item(1, "chain_structure", h["id"]))
    t = _terminal(h)
    if not any(s.get("explained_by") == t for s in concl.get("symptoms", [])):
        items.append(_item(1, "chain_does_not_reach_symptoms", t))
    return items


def _relation(x: _Ctx, ca: dict, ea: dict) -> tuple[str, float | None]:
    hc, he = ca.get("host"), ea.get("host")
    src_c, src_e = x.single_source(ca["evidence_id"]), x.single_source(ea["evidence_id"])
    if hc is None and he is None and src_c and src_c == src_e:
        x.flags.append(
            _item(2, "clocks_assumed_shared", None, relation="same_source_null_host", source=src_c)
        )
        return "same_source_null_host", 0.0
    if hc is None or he is None:
        return "clock_unknown", None
    if hc == he:
        if src_c and src_c == src_e:
            return "same_clock", 0.0
        x.flags.append(
            _item(2, "clocks_assumed_shared", None, relation="same_host_cross_source", host=hc)
        )
        return "same_host_cross_source", 0.0
    row = x.pairs.get(tuple(sorted([hc, he])))
    if row is None:
        return "not_estimable", None
    if _isnan(row.get("offset_ms")) or _isnan(row.get("residual_std_ms")):
        return "residual_unknown", None
    return "measured", abs(row["offset_ms"]) / 1000.0 + x.inp.residual_k * row[
        "residual_std_ms"
    ] / 1000.0


def _c2(x: _Ctx, h: dict, concl: dict) -> list[dict]:
    items = []
    noted = {
        n["link_id"]
        for n in concl.get("ordering_notes", [])
        if all(x.survives(e) for e in n["evidence_ids"])
    }
    links = _links_of(x.st, h)
    if not links:
        return [_item(2, "no_links", h["id"])]
    for lid in links:
        lk = x.st.links[lid]
        ca, ea = lk.get("cause_at"), lk.get("effect_at")
        if not ca or not ea:
            items.append(_item(2, "deferred" if lid in noted else "anchor_missing", lid))
            continue
        delta = ea["ts_s"] - ca["ts_s"]
        g = max(x.gran(ca["evidence_id"]), x.gran(ea["evidence_id"]))
        rel, u = _relation(x, ca, ea)
        det = {
            "delta_s": round(delta, 6),
            "granularity_s": g,
            "relation": rel,
            "pair": [ca.get("host"), ea.get("host")],
        }
        if u is None:
            items.append(
                _item(2, "deferred" if lid in noted else "ordering_unmeasured_clock", lid, **det)
            )
            continue
        bound = g + u
        det["bound_s"] = round(bound, 6)
        if delta >= bound:
            continue
        if delta <= -bound:
            items.append(_item(2, "timestamps_reversed", lid, **det))
        else:
            items.append(
                _item(2, "deferred" if lid in noted else "ordering_within_uncertainty", lid, **det)
            )
    return items


def _c3(x: _Ctx, h: dict) -> list[dict]:
    items = []
    for b in h["branches"]:
        if not b["links"]:
            continue
        first = x.st.links[b["links"][0]]
        if not any(x.contrast_ok(e) for e in first["evidence"]):
            items.append(_item(3, "no_contrast_for_initiating_condition", b["links"][0]))
    if not h["branches"]:
        items.append(_item(3, "no_links", h["id"]))
    return items


def _active(h: dict, stance: str) -> set[str]:
    return {
        e["evidence_id"] for e in h["evidence"] if e["stance"] == stance and e["withdrawn"] is None
    }


def _refutation_items(x: _Ctx, a: dict | None) -> list[dict]:
    items = []
    against_a = _active(a, "contradicts") if a else set()
    for h in x.st.hypotheses.values():
        if h["status"] != "refuted":
            continue
        basis = (h["status_basis"] or {}).get("evidence_ids", [])
        surv = [e for e in basis if x.survives(e)]
        if not surv:
            items.append(_item(4, "refutation_evidence_excluded", h["id"]))
        elif a is not None and all(e in against_a for e in surv):
            items.append(_item(4, "refutation_not_discriminating", h["id"]))
    return items


def _c4(x: _Ctx, a: dict | None) -> list[dict]:
    st = x.st
    items = []
    if len(st.accepted()) != 1:
        items.append(_item(4, "accepted_count", None, accepted=st.accepted()))
    live = sorted(h["id"] for h in st.hypotheses.values() if h["status"] == "live")
    if live:
        items.append(_item(4, "live_hypothesis_remaining", None, ids=live))
    ok, detail = two_live_check(st)
    if not ok:
        items.append(_item(4, "too_few_clusters", None, **detail))
    items += _refutation_items(x, a)
    for h in st.hypotheses.values():
        if h["status"] == "unresolved" and h["origin"] != "scaffold":
            sup = [e for e in _active(h, "supports") if x.survives(e)]
            con = [e for e in _active(h, "contradicts") if x.survives(e)]
            if sup and not con:
                x.flags.append(_item(4, "supported_unresolved_competitor", h["id"]))
    return items


def _c5(x: _Ctx, concl: dict, allowed_targets: set[str]) -> list[dict]:
    items = []
    syms = concl.get("symptoms", [])
    if not syms:
        return [_item(5, "no_symptoms_listed")]
    for i, s in enumerate(syms):
        surv = [e for e in s["evidence_ids"] if x.survives(e)]
        if not surv:
            items.append(_item(5, "symptom_without_evidence", None, symptom=i))
        elif all(x.st.evidence[e]["evidence_kind"] == "claimed_content" for e in surv):
            x.flags.append(_item(5, "symptom_claimed_content_only", None, symptom=i))
        if s.get("explained_by") and s["explained_by"] not in allowed_targets:
            items.append(_item(5, "symptom_explained_by_unknown", s["explained_by"], symptom=i))
    return items


def _c6(x: _Ctx) -> tuple[str, list[dict]]:
    if x.inp.critic_rounds_done == 0:
        return "not_applicable", []
    open_ = sorted(o["id"] for o in x.st.objections.values() if o["status"] == "open")
    return ("unmet", [_item(6, "objection_open", None, ids=open_)]) if open_ else ("met", [])


def _touches(x: _Ctx, fa: dict, lk: dict) -> bool:
    fcov = [c for c in x.cov_by_q.get(fa["query_id"], []) if c["kind"] == "attempt"]
    if not fcov:
        return False
    evids = {e["evidence_id"] for e in lk["evidence"]}
    for end in ("cause_at", "effect_at"):
        if lk.get(end):
            evids.add(lk[end]["evidence_id"])
    for eid in evids:
        if eid not in x.st.evidence:
            continue
        s = x.scope(x.st.evidence[eid]["query_id"])
        for fc in fcov:
            fs = set(fc["sources"] or [])
            if not (fs & s["sources"] or "unknown" in fs or "unknown" in s["sources"]):
                continue
            ent_ok = True
            for kind, vals in (fc.get("requested") or {}).items():
                if (
                    isinstance(vals, list)
                    and kind in s["entities"]
                    and not set(map(str, vals)) & s["entities"][kind]
                ):
                    ent_ok = False
            if not ent_ok:
                continue
            fw = fc.get("requested_window")
            if fw is None or any(
                w is None or (fw["start_s"] < w["end_s"] and w["start_s"] < fw["end_s"])
                for w in s["windows"] or [None]
            ):
                return True
    return False


def _c8(x: _Ctx, h: dict) -> list[dict]:
    items = []
    unresolved = [
        f
        for f in x.st.failures.values()
        if f["resolution"] is None and f["error_type"] != "invalid_query"
    ]
    for lid in _links_of(x.st, h):
        lk = x.st.links[lid]
        non_claim = [
            e["evidence_id"]
            for e in lk["evidence"]
            if x.survives(e["evidence_id"])
            and x.st.evidence[e["evidence_id"]]["evidence_kind"] != "claimed_content"
        ]
        if not non_claim:
            items.append(_item(8, "claimed_content_only", lid))
        elif all(not x.guard_ok(e) for e in non_claim):
            x.flags.append(_item(8, "unverified_scope_only", lid))
        for fa in unresolved:
            if _touches(x, fa, lk):
                items.append(
                    _item(
                        8,
                        "unresolved_failure_touches_link",
                        lid,
                        failure=fa["failure_id"],
                        tool=fa["tool"],
                    )
                )
    return items


def _c11(x: _Ctx, h: dict) -> tuple[str, list[dict]]:
    if len(h["branches"]) < 2:
        return "not_applicable", []
    cs = []
    for b in h["branches"]:
        cs.append(
            {
                e["evidence_id"]
                for lid in b["links"]
                for e in x.st.links[lid]["evidence"]
                if x.contrast_ok(e)
            }
        )
    items = []
    for i, c in enumerate(cs):
        others = set().union(*(cs[:i] + cs[i + 1 :]))
        if not c - others:
            items.append(_item(11, "branch_contrast_not_distinct", h["branches"][i]["id"]))
    return ("unmet" if items else "met"), items


def _c10(x: _Ctx) -> list[dict]:
    return [
        _item(10, "human_hypothesis_undispositioned", h["id"])
        for h in x.st.hypotheses.values()
        if h["origin"] == "human" and h["status"] == "live"
    ]


def _gaps(x: _Ctx, cited: set[str], anchor_ts: list[float]) -> dict:
    chain_src, chain_ent, times = set(), {}, list(anchor_ts)
    for eid in cited:
        if eid not in x.st.evidence:
            continue
        e = x.st.evidence[eid]
        s = x.scope(e["query_id"])
        chain_src |= s["sources"]
        for k, v in s["entities"].items():
            chain_ent.setdefault(k, set()).update(v)
        if e.get("time_range"):
            times += [e["time_range"]["start_s"], e["time_range"]["end_s"]]
    w = (min(times), max(times)) if times else None
    groups: dict[tuple, dict] = {}
    for c in x.inp.coverage:
        un = c.get("unobserved_entities") or {}
        reason = c.get("attempt_reason")
        if reason not in ("entity_absent_from_source", "join_key_unpopulated") and not (
            c["kind"] == "queried" and un
        ):
            continue
        reason = (
            reason
            if reason in ("entity_absent_from_source", "join_key_unpopulated")
            else "unobserved_in_successful_query"
        )
        srcs = set(c["sources"] or [])
        rw = c.get("requested_window")
        overlap = w is None or rw is None or (rw["start_s"] <= w[1] and w[0] <= rw["end_s"])
        ents = dict(un)
        for k, v in (c.get("requested") or {}).items():
            if isinstance(v, list) and reason != "unobserved_in_successful_query":
                ents.setdefault(k, v)
        shares = bool(srcs & chain_src) or any(
            set(map(str, v)) & chain_ent.get(k, set()) for k, v in ents.items()
        )
        bucket = "on_path" if overlap and shares else "other"
        for src in sorted(srcs) or ["unknown"]:
            for kind, vals in sorted(ents.items()) or [("", [])]:
                key = (bucket, reason, src, kind, ",".join(map(str, vals)))
                g = groups.setdefault(
                    key,
                    {
                        "reason": reason,
                        "source": src,
                        "kind": kind or None,
                        "values": list(vals),
                        "query_ids": [],
                    },
                )
                g["query_ids"].append(c["query_id"])
    out = {"on_path": [], "other": []}
    for key in sorted(groups):
        out[key[0]].append(groups[key])
    return out


def _cited(x: _Ctx, hyps: list[dict], concl: dict) -> tuple[set[str], list[float]]:
    cited, ts = set(), []
    for h in hyps:
        for lid in _links_of(x.st, h):
            lk = x.st.links[lid]
            cited |= {e["evidence_id"] for e in lk["evidence"]}
            for end in ("cause_at", "effect_at"):
                if lk.get(end):
                    cited.add(lk[end]["evidence_id"])
                    ts.append(lk[end]["ts_s"])
        for fid in h["factor_ids"]:
            cited |= {e["evidence_id"] for e in x.st.factors[fid]["evidence"]}
    for s in concl.get("symptoms", []):
        cited |= set(s["evidence_ids"])
    for n in concl.get("ordering_notes", []):
        cited |= set(n["evidence_ids"])
    return {c for c in cited if x.survives(c) or x.inp.phase != "final"}, ts


def verifier_scope(state: LedgerState, conclusion: dict | None) -> list[str]:
    """design-d6-d7 §11 (T1): the evidence the deterministic verifier must
    re-execute: the accepted chain (or every candidate's), factors,
    refutation bases, symptoms, ordering notes and objection resolutions."""
    ids: set[str] = set()
    if conclusion is None:
        return []
    aid = conclusion.get("accepted_hypothesis_id")
    hyps = (
        [state.hypotheses[aid]]
        if aid
        else [
            h
            for h in state.hypotheses.values()
            if h["status"] in ("unresolved", "live", "accepted") and h["origin"] != "scaffold"
        ]
    )
    for h in hyps:
        for lid in _links_of(state, h):
            lk = state.links[lid]
            ids |= {e["evidence_id"] for e in lk["evidence"]}
            for end in ("cause_at", "effect_at"):
                if lk.get(end):
                    ids.add(lk[end]["evidence_id"])
        for fid in h["factor_ids"]:
            ids |= {e["evidence_id"] for e in state.factors[fid]["evidence"]}
    for h in state.hypotheses.values():
        if h["status"] == "refuted":
            ids |= set((h["status_basis"] or {}).get("evidence_ids", []))
    for s in conclusion.get("symptoms", []):
        ids |= set(s["evidence_ids"])
    for n in conclusion.get("ordering_notes", []):
        ids |= set(n["evidence_ids"])
    for o in state.objections.values():
        for r in o["resolutions"]:
            ids |= set(r.get("evidence_ids") or [])
    return sorted(i for i in ids if i in state.evidence)


def _status(items: list[dict]) -> str:
    codes = [i["code"] for i in items]
    if any(c != "deferred" for c in codes):
        return "unmet"
    return "deferred" if codes else "met"


def _candidates(x: _Ctx, concl: dict) -> list[dict]:
    """§8 ranking: tiers by criteria {1,2,3,8,11,C0} met as if accepted; ties labelled."""
    st = x.st
    cands = [
        h
        for h in st.hypotheses.values()
        if h["origin"] != "scaffold" and h["status"] in ("unresolved", "live", "accepted")
    ]
    for h in st.hypotheses.values():
        if h["status"] == "refuted" and h["origin"] != "scaffold":
            basis = (h["status_basis"] or {}).get("evidence_ids", [])
            if basis and not any(x.survives(e) for e in basis):
                cands.append(h)
                x.flags.append(_item(4, "refutation_failed_verification", h["id"]))
    rows = []
    saved = list(x.flags)
    for h in sorted(cands, key=lambda h: h["id"]):
        met = []
        met.append(
            _status(
                _c1(
                    x,
                    h,
                    dict(
                        concl,
                        symptoms=[
                            dict(s, explained_by=_terminal(h)) for s in concl.get("symptoms", [])
                        ]
                        or [{"explained_by": _terminal(h), "evidence_ids": []}],
                    ),
                )
            )
            != "unmet"
        )
        met.append(_status(_c2(x, h, concl)) != "unmet")
        met.append(_status(_c3(x, h)) != "unmet")
        met.append(_status(_c8(x, h)) != "unmet")
        s11, i11 = _c11(x, h)
        met.append(s11 != "unmet")
        met.append(not any(x.survives(e) for e in _active(h, "contradicts")))
        rows.append(
            {
                "hypothesis_id": h["id"],
                "cluster_id": h["cluster_id"],
                "criteria_met": sum(met),
                "of": len(met),
            }
        )
    x.flags = saved
    rows.sort(key=lambda r: (-r["criteria_met"], r["hypothesis_id"]))
    tiers: dict[int, int] = {}
    for r in rows:
        tiers[r["criteria_met"]] = tiers.get(r["criteria_met"], 0) + 1
    for r in rows:
        r["tied"] = tiers[r["criteria_met"]] > 1
    standing = st.standing_id()
    if standing:
        rows.append(
            {
                "hypothesis_id": standing,
                "standing": True,
                "status": st.hypotheses[standing]["status"],
                "note": "listed separately; not ranked (it has no chain by construction)",
            }
        )
    return rows


def done_check(inp: DoneCheckInput) -> DoneCheckResult:
    x = _Ctx(inp)
    st = inp.state
    concl = st.latest_conclusion()
    crit: dict[str, dict] = {}
    if concl is None:
        item = _item(0, "no_conclusion_proposed")
        return DoneCheckResult(inp.phase, None, None, False, {}, [item])
    kind = concl["kind"]
    aid = concl.get("accepted_hypothesis_id")
    a = st.hypotheses.get(aid) if aid else None

    def put(n: int, items: list[dict], status: str | None = None):
        crit[str(n)] = {"status": status or _status(items), "items": items}

    s6, i6 = _c6(x)
    put(6, i6, s6)
    put(10, _c10(x))
    if kind == "conclusive" and a is not None:
        put(1, _c1(x, a, concl))
        put(2, _c2(x, a, concl))
        put(3, _c3(x, a))
        put(4, _c4(x, a))
        targets = set(_links_of(st, a)) | set(a["factor_ids"])
        put(5, _c5(x, concl, targets))
        put(8, _c8(x, a))
        s11, i11 = _c11(x, a)
        put(11, i11, s11)
        i9 = []
        if not concl.get("sufficiency_note"):
            i9.append(_item(9, "sufficiency_note_missing"))
        if not concl.get("confidence_basis"):
            i9.append(_item(9, "confidence_basis_missing"))
        put(9, i9)
        hyps_for_gaps = [a]
    else:
        # inconclusive check (§8)
        items4 = []
        if st.accepted():
            items4.append(_item(4, "accepted_in_inconclusive", None, ids=st.accepted()))
        live = sorted(h["id"] for h in st.hypotheses.values() if h["status"] == "live")
        if live:
            items4.append(_item(4, "live_hypothesis_remaining", None, ids=live))
        if not any(h["status"] == "unresolved" for h in st.hypotheses.values()):
            items4.append(_item(4, "no_unresolved_candidate"))
        ok, detail = two_live_check(st)
        if not ok:
            items4.append(_item(4, "too_few_clusters", None, **detail))
        items4 += _refutation_items(x, None)
        cand_h = [
            h
            for h in st.hypotheses.values()
            if h["origin"] != "scaffold" and h["status"] in ("unresolved", "live", "accepted")
        ]
        clusters = {h["cluster_id"] for h in cand_h}
        if len(clusters) >= 2:
            named = {hid for d in concl.get("discriminators", []) for hid in d["between"]}
            for h in cand_h:
                if h["id"] not in named:
                    items4.append(_item(4, "candidate_without_discriminator", h["id"]))
        put(4, items4)
        targets = {t for h in cand_h for t in _links_of(st, h) + h["factor_ids"]}
        put(5, _c5(x, concl, targets))
        for n in (1, 2, 3, 8, 11):
            crit[str(n)] = {"status": "not_applicable", "items": []}
        put(9, [])
        hyps_for_gaps = cand_h
    # criterion 7 (final only)
    if inp.phase == "final":
        i7 = []
        scope = verifier_scope(st, concl)
        for eid in scope:
            if eid not in st.verdicts:
                i7.append(_item(7, "verdict_missing", eid))
            elif st.verdicts[eid].get("verdict") == "fail" and eid not in x.stripped:
                x.stripped.append(eid)
        if a is not None:
            for lid in _links_of(st, a):
                ent = st.entailments.get(lid)
                if ent and ent.get("verdict") == "does_not_support":
                    i7.append(_item(7, "entailment_rejected", lid))
                elif ent and ent.get("verdict") == "insufficient":
                    x.flags.append(_item(7, "entailment_insufficient", lid))
        for n in ("1", "2", "3", "5", "8", "11"):
            if crit.get(n, {}).get("status") == "unmet":
                i7.append(_item(7, "criterion_unmet_after_verification", None, criterion=int(n)))
        put(7, i7)
    else:
        crit["7"] = {"status": "not_applicable", "items": []}
    # guarded-evidence flag
    for eid, e in st.evidence.items():
        if e.get("guard") == "unverified_scope":
            x.flags.append(_item(2, "unverified_scope_evidence", eid))
    # strong claims (9a)
    if kind == "conclusive" and concl.get("confidence_basis"):
        flagged = {f["criterion"] for f in x.flags}
        for n in concl["confidence_basis"].get("strong", []):
            c = crit.get(str(n), {})
            if c.get("status") != "met" or n in flagged:
                crit["9"]["items"].append(
                    _item(9, "strong_claim_not_met", None, criterion=n, status=c.get("status"))
                )
        crit["9"]["status"] = _status(crit["9"]["items"])
    cited, ts = _cited(x, hyps_for_gaps, concl)
    gaps = _gaps(x, cited, ts)
    candidates = _candidates(x, concl) if kind == "inconclusive" else None
    gating = (
        set(range(1, 12))
        if inp.phase == "final"
        else GATING_PRE | ({6} if inp.critic_rounds_done else set())
    )
    unmet = [
        i
        for n, c in sorted(crit.items(), key=lambda kv: int(kv[0]))
        if int(n) in gating
        for i in c["items"]
        if i["code"] != "deferred" and c["status"] == "unmet"
    ]
    ok = not unmet
    offsets_sha = hashlib.sha256(
        json.dumps(inp.clock_offsets, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return DoneCheckResult(
        phase=inp.phase,
        kind=kind,
        conclusion_seq=concl.get("seq"),
        ok=ok,
        criteria=crit,
        unmet=unmet,
        flags=sorted(x.flags, key=lambda f: (f["criterion"], f["code"], str(f["target"]))),
        stripped=sorted(x.stripped),
        observability_gaps=gaps,
        candidates=candidates,
        inputs={
            "ledger_seq_max": st.last_seq,
            "clock_offsets_sha256": offsets_sha,
            "residual_k": inp.residual_k,
        },
    )


# ---- lead-facing rendering (fixed, domain-free templates; part of the prompt audit) ------

TEMPLATES = {
    "no_conclusion_proposed": "no conclusion has been proposed",
    "no_links": "{target}: the hypothesis has no causal chain yet",
    "link_without_evidence": "{target}: the link cites no surviving evidence",
    "anchor_evidence_excluded": "{target}: an anchor's evidence did not survive verification",
    "chain_structure": "{target}: exactly one branch must end at the symptoms",
    "chain_does_not_reach_symptoms": "{target}: no listed symptom is explained by the chain's final link",
    "anchor_missing": "{target}: pin cause_at and effect_at to cited rows, or add an ordering note citing non-timestamp evidence",
    "ordering_unmeasured_clock": "{target}: no measured clock relation between {pair} ({relation}); ordering can't rest on timestamps alone. Add ordering_notes for this link citing non-timestamp evidence, or anchor both ends on one host",
    "ordering_within_uncertainty": "{target}: delta {delta_s}s is inside the uncertainty bound {bound_s}s ({relation} {pair}). Pin anchors further apart, pin both to one host, or add an ordering note",
    "timestamps_reversed": "{target}: the effect's timestamp precedes the cause's by more than the uncertainty bound ({delta_s}s vs {bound_s}s); re-pin the anchors",
    "no_contrast_for_initiating_condition": "{target}: the first link of this branch needs contrast-role evidence (a baseline, an unaffected peer or class) that is verified and not scope-unverified",
    "accepted_count": "exactly one hypothesis must be accepted for a conclusive result",
    "live_hypothesis_remaining": "hypotheses still live: {ids}; refute each with cited contradicting evidence or mark it unresolved",
    "too_few_clusters": "fewer than two distinct explanations were ever considered (rewordings count once)",
    "refutation_evidence_excluded": "{target}: its refutation evidence did not survive verification",
    "refutation_not_discriminating": "{target}: every surviving refutation evidence also contradicts the accepted hypothesis, so it separates neither",
    "no_symptoms_listed": "list the symptoms the conclusion explains",
    "symptom_without_evidence": "symptom #{symptom} cites no surviving evidence",
    "symptom_explained_by_unknown": "symptom #{symptom}: explained_by {target} is not a link or factor of the concluded hypothesis",
    "objection_open": "open critic objections: {ids}",
    "claimed_content_only": "{target}: the link rests only on what log lines claim; add occurrence or derived evidence",
    "unresolved_failure_touches_link": "{target}: tool failure {failure} ({tool}) overlaps this link's evidence scope; retry and resolve it, or cite evidence outside that scope",
    "sufficiency_note_missing": "a conclusive result needs a sufficiency_note addressing the data-sufficiency declaration",
    "confidence_basis_missing": "a conclusive result needs a confidence_basis",
    "strong_claim_not_met": "criterion {criterion} is listed as strongly met but is {status}",
    "human_hypothesis_undispositioned": "{target}: a human-supplied hypothesis is still live; accept, refute with evidence, or mark unresolved",
    "branch_contrast_not_distinct": "{target}: each branch needs contrast evidence of its own, not shared with another branch",
    "accepted_in_inconclusive": "an inconclusive result can't have an accepted hypothesis: {ids}",
    "no_unresolved_candidate": "an inconclusive result carries at least one hypothesis as unresolved",
    "candidate_without_discriminator": "{target}: name the evidence that would discriminate it from the other candidates",
    "verdict_missing": "{target}: not yet verified",
    "entailment_rejected": "{target}: the cited data was judged not to support the link",
    "criterion_unmet_after_verification": "criterion {criterion} is unmet once unverified evidence is removed",
}


def render_unmet(result: DoneCheckResult, short=lambda s: s, cap: int = 20) -> list[str]:
    lines = []
    for it in result.unmet[:cap]:
        fields = {
            k: (short(v) if isinstance(v, str) and "/" in v else v) for k, v in it["detail"].items()
        }
        fields["target"] = short(it["target"]) if it["target"] else ""
        if "ids" in fields:
            fields["ids"] = [short(i) for i in fields["ids"]]
        try:
            text = TEMPLATES.get(it["code"], it["code"]).format(**fields)
        except (KeyError, IndexError):
            text = it["code"]
        lines.append(f"C{it['criterion']} {it['code']}: {text}")
    if len(result.unmet) > cap:
        lines.append(f"{len(result.unmet) - cap} more unmet items")
    return lines


# ---- adapter: assemble the input from a run, record the result ------------------------------


def load_coverage(con) -> list[dict]:
    cur = con.execute("SELECT * FROM coverage ORDER BY coverage_id")
    cols = [d[0] for d in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r, strict=True))
        for k in (
            "sources",
            "requested",
            "requested_window",
            "returned",
            "returned_window",
            "unobserved_entities",
        ):
            d[k] = json.loads(d[k]) if d.get(k) else None
        out.append(d)
    return out


def load_queries(con) -> dict[str, dict]:
    return {
        r[0]: {"tool": r[1], "input_query_ids": json.loads(r[2] or "[]"), "status": r[3]}
        for r in con.execute(
            "SELECT query_id, tool, input_query_ids, status FROM queries"
        ).fetchall()
    }


def run_coverage(ledger) -> tuple[list[dict], dict[str, dict]]:
    """Coverage and query rows for this run and its lineage (read-only)."""
    from agentic_rca.ledger.rundb import connect_run

    coverage, queries = load_coverage(ledger.con), load_queries(ledger.con)
    for _rid, path in ledger.lineage:
        pcon = connect_run(path, read_only=True)
        coverage += load_coverage(pcon)
        queries.update(load_queries(pcon))
        pcon.close()
    return coverage, queries


def precheck_for(
    ledger,
    *,
    clock_offsets: list[dict],
    source_ts_unit_s: dict[str, float],
    residual_k: float = 2.0,
    rounds=lambda: 0,
):
    """A callable(phase) for the lead loop: runs the check, records a
    `done_check_result` event, returns the lead-facing summary."""

    def check(phase: str) -> dict:
        # in-loop requests are always pre_challenge; the orchestrator runs `final`
        phase = "final" if phase == "final" else "pre_challenge"
        coverage, queries = run_coverage(ledger)
        res = done_check(
            DoneCheckInput(
                ledger.state(),
                coverage,
                queries,
                clock_offsets,
                source_ts_unit_s,
                phase,
                rounds(),
                residual_k,
            )
        )
        payload = asdict(res)
        ledger.apply("done_check_result", payload, actor="scaffold")
        return {
            "met": res.ok,
            "unmet": render_unmet(res, ledger.short),
            "flags": [
                f"{f['code']} {ledger.short(f['target']) if f['target'] else ''}".strip()
                for f in res.flags
            ][:20],
            "result": payload,
        }

    return check
