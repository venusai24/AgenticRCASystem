"""`ledger_view`: the compact, authoritative ledger render (§4.3).

Stable ordering (by id), a hard size ceiling, and `hide_origin` for the
critic, who must not see where a hypothesis came from (§4.4).
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_rca.ledger.fold import LedgerState, two_live_check


def render(
    state: LedgerState,
    short: Callable[[str], str],
    *,
    hide_origin: bool = False,
    sections: tuple[str, ...] = (
        "hypotheses",
        "evidence",
        "questions",
        "objections",
        "failures",
        "focus",
    ),
    max_chars: int = 12000,
) -> str:
    s = short
    out: list[str] = []
    ok, detail = two_live_check(state)
    out.append(
        f"[>=2-live: {'ok' if ok else 'NOT MET'}; clusters standing={detail['clusters_standing']} "
        f"eliminated={detail['clusters_eliminated']}"
        + (f"; dedup pending={[s(x) for x in detail['pending']]}" if detail["pending"] else "")
        + (
            f"; dedup unchecked={[s(x) for x in detail['unchecked']]}"
            if detail["unchecked"]
            else ""
        )
        + "]"
    )
    if "hypotheses" in sections:
        out.append("HYPOTHESES")
        for hid in sorted(state.hypotheses):
            h = state.hypotheses[hid]
            origin = "" if hide_origin else f" origin={h['origin']}"
            out.append(
                f"- {s(hid)} [{h['status']}] cluster={s(h['cluster_id'])}"
                f"({h['cluster_status']}){origin}: {h['statement']}"
            )
            for b in h["branches"]:
                joins = f" -> joins {s(b['joins'])}" if b["joins"] else " -> symptoms"
                out.append(f"    branch {s(b['id'])}{joins}")
                for lid in b["links"]:
                    lk = state.links[lid]
                    ev = ", ".join(f"{s(e['evidence_id'])}:{e['role']}" for e in lk["evidence"])
                    anchors = ""
                    if lk.get("cause_at") and lk.get("effect_at"):
                        c, e = lk["cause_at"], lk["effect_at"]
                        anchors = f" @ {c['ts_s']}({c['host']}) -> {e['ts_s']}({e['host']})"
                    out.append(f"      {s(lid)}: {lk['cause']} => {lk['effect']} [{ev}]{anchors}")
            for fid in h["factor_ids"]:
                f = state.factors[fid]
                out.append(f"    factor {s(fid)}: {f['statement']}")
            active = [e for e in h["evidence"] if e["withdrawn"] is None]
            if active:
                out.append(
                    "    evidence: "
                    + ", ".join(f"{s(e['evidence_id'])}({e['stance']})" for e in active)
                )
            for pr in h["predictions"]:
                out.append(
                    f"    prediction {s(pr['prediction_id'])} [{pr['status']}]: {pr['text']}"
                )
            if h["status_basis"]:
                b = h["status_basis"]
                same = f" same_as={s(b['same_as'])}" if b.get("same_as") else ""
                out.append(
                    f"    basis: {b['reason']} [{', '.join(s(x) for x in b['evidence_ids'])}]{same}"
                )
    if "evidence" in sections and state.evidence:
        out.append("EVIDENCE")
        for eid in sorted(state.evidence):
            e = state.evidence[eid]
            flags = []
            if e.get("guard") == "unverified_scope":
                flags.append("UNVERIFIED-SCOPE")
            if e["evidence_kind"] != "occurrence":
                flags.append(e["evidence_kind"])
            v = state.verdicts.get(eid)
            if v:
                flags.append(f"verdict={v['verdict']}")
            rows = ",".join(str(r["row"]) for r in e["row_refs"]) or "none(0 rows)"
            f = f" [{' '.join(flags)}]" if flags else ""
            out.append(f"- {s(eid)} q={s(e['query_id'])} rows={rows}{f}: {e['claim']}")
    if "questions" in sections:
        open_q = [q for q in state.questions.values() if q["status"] == "open"]
        if open_q:
            out.append("OPEN QUESTIONS")
            out += [f"- {s(q['id'])}: {q['text']}" for q in sorted(open_q, key=lambda q: q["id"])]
    if "objections" in sections and state.objections:
        out.append("OBJECTIONS")
        for oid in sorted(state.objections):
            o = state.objections[oid]
            tgt = s(o["target"]["link_id"] or o["target"]["hypothesis_id"])
            out.append(f"- {s(oid)} r{o['round']} [{o['status']}] on {tgt}: {o['text']}")
            for r in o["resolutions"]:
                out.append(f"    {r['kind']}: {r.get('note', '')}")
    if "failures" in sections:
        unresolved = [f for f in state.failures.values() if f["resolution"] is None]
        if unresolved:
            out.append("UNRESOLVED TOOL FAILURES")
            out += [
                f"- {s(f['failure_id'])} {f['tool']} q={s(f['query_id'])} {f['error_type']}: {f['diagnostic']}"
                for f in sorted(unresolved, key=lambda f: f["failure_id"])
            ]
    if "focus" in sections and state.focus:
        fo = state.focus
        out.append(
            f"FOCUS: {fo['window']['start_s']}..{fo['window']['end_s']} because {fo['reason']}"
        )
    text = "\n".join(out)
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit("\n", 1)[0]
        hidden = text[len(cut) :].count("\n")
        text = cut + f"\n[... {hidden} more lines; call ledger_view with a section filter]"
    return text
