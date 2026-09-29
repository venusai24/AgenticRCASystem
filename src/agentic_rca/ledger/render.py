"""`ledger_view`: the compact, authoritative ledger render (§4.3).

Stable ordering (by id), a hard size ceiling, and `hide_origin` for the
critic, who must not see where a hypothesis came from (§4.4).

Truncation is priority-based: P1 (focus, open objections, live/accepted
hypotheses, open questions) is always kept; P2-P4 are appended while
budget remains, with a semantic omission hint when items are dropped.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_rca.ledger.fold import LedgerState, two_live_check


def _join(lines: list[str]) -> str:
    return "\n".join(lines)


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

    # ── P1: two-live header (always present, not in sections filter) ─────────
    ok, detail = two_live_check(state)
    p1: list[str] = [
        f"[>=2-live: {'ok' if ok else 'NOT MET'}; clusters standing={detail['clusters_standing']} "
        f"eliminated={detail['clusters_eliminated']}"
        + (f"; dedup pending={[s(x) for x in detail['pending']]}" if detail["pending"] else "")
        + (
            f"; dedup unchecked={[s(x) for x in detail['unchecked']]}"
            if detail["unchecked"]
            else ""
        )
        + "]"
    ]

    # ── Focus ─────────────────────────────────────────────────────────────────
    if "focus" in sections and state.focus:
        fo = state.focus
        p1.append(
            f"FOCUS: {fo['window']['start_s']}..{fo['window']['end_s']} because {fo['reason']}"
        )

    # ── Open objections ───────────────────────────────────────────────────────
    if "objections" in sections and state.objections:
        open_obj = [o for o in state.objections.values() if o["status"] == "open"]
        if open_obj:
            p1.append("OBJECTIONS (open)")
            for o in sorted(open_obj, key=lambda o: o["id"]):
                tgt = s(o["target"]["link_id"] or o["target"]["hypothesis_id"])
                p1.append(f"- {s(o['id'])} r{o['round']} [open] on {tgt}: {o['text']}")

    # ── Live / accepted hypotheses ────────────────────────────────────────────
    live_h_lines: list[str] = []
    refuted_h_lines: list[str] = []
    if "hypotheses" in sections:
        live_statuses = {"live", "accepted"}
        live_h_lines.append("HYPOTHESES")
        for hid in sorted(state.hypotheses):
            h = state.hypotheses[hid]
            origin = "" if hide_origin else f" origin={h['origin']}"
            header = (
                f"- {s(hid)} [{h['status']}] cluster={s(h['cluster_id'])}"
                f"({h['cluster_status']}){origin}: {h['statement']}"
            )
            body: list[str] = []
            for b in h["branches"]:
                joins = f" -> joins {s(b['joins'])}" if b["joins"] else " -> symptoms"
                body.append(f"    branch {s(b['id'])}{joins}")
                for lid in b["links"]:
                    lk = state.links[lid]
                    ev = ", ".join(f"{s(e['evidence_id'])}:{e['role']}" for e in lk["evidence"])
                    anchors = ""
                    if lk.get("cause_at") and lk.get("effect_at"):
                        c, e = lk["cause_at"], lk["effect_at"]
                        anchors = f" @ {c['ts_s']}({c['host']}) -> {e['ts_s']}({e['host']})"
                    body.append(f"      {s(lid)}: {lk['cause']} => {lk['effect']} [{ev}]{anchors}")
            for fid in h["factor_ids"]:
                f = state.factors[fid]
                body.append(f"    factor {s(fid)}: {f['statement']}")
            active = [e for e in h["evidence"] if e["withdrawn"] is None]
            if active:
                body.append(
                    "    evidence: "
                    + ", ".join(f"{s(e['evidence_id'])}({e['stance']})" for e in active)
                )
            for pr in h["predictions"]:
                body.append(
                    f"    prediction {s(pr['prediction_id'])} [{pr['status']}]: {pr['text']}"
                )
            if h["status_basis"]:
                b = h["status_basis"]
                same = f" same_as={s(b['same_as'])}" if b.get("same_as") else ""
                body.append(
                    f"    basis: {b['reason']} [{', '.join(s(x) for x in b['evidence_ids'])}]{same}"
                )

            if h["status"] in live_statuses:
                live_h_lines.append(header)
                live_h_lines.extend(body)
            else:
                refuted_h_lines.append(header)
                refuted_h_lines.extend(body)

        p1.extend(live_h_lines)

    # ── Open questions ────────────────────────────────────────────────────────
    if "questions" in sections:
        open_q = [q for q in state.questions.values() if q["status"] == "open"]
        if open_q:
            p1.append("OPEN QUESTIONS")
            p1 += [f"- {s(q['id'])}: {q['text']}" for q in sorted(open_q, key=lambda q: q["id"])]

    # ── P2: unresolved failures + resolved/conceded objections ────────────────
    p2: list[str] = []
    if "failures" in sections:
        unresolved = [f for f in state.failures.values() if f["resolution"] is None]
        if unresolved:
            p2.append("UNRESOLVED TOOL FAILURES")
            p2 += [
                f"- {s(f['failure_id'])} {f['tool']} q={s(f['query_id'])} {f['error_type']}: {f['diagnostic']}"
                for f in sorted(unresolved, key=lambda f: f["failure_id"])
            ]

    if "objections" in sections and state.objections:
        closed_obj = [
            o for o in state.objections.values() if o["status"] in ("resolved", "conceded")
        ]
        if closed_obj:
            p2.append("OBJECTIONS (resolved/conceded)")
            for o in sorted(closed_obj, key=lambda o: o["id"])[-5:]:  # ponytail: last 5 only
                tgt = s(o["target"]["link_id"] or o["target"]["hypothesis_id"])
                p2.append(f"- {s(o['id'])} r{o['round']} [{o['status']}] on {tgt}: {o['text']}")
                for r in o["resolutions"]:
                    p2.append(f"    {r['kind']}: {r.get('note', '')}")

    # ── P3: evidence cited by live/accepted hypotheses ────────────────────────
    p3: list[str] = []
    if "evidence" in sections and state.evidence:
        live_hids = {
            hid for hid, h in state.hypotheses.items() if h["status"] in ("live", "accepted")
        }
        cited_eids: set[str] = set()
        for hid in live_hids:
            h = state.hypotheses[hid]
            for e in h["evidence"]:
                if e["withdrawn"] is None:
                    cited_eids.add(e["evidence_id"])

        cited_ev = {eid: state.evidence[eid] for eid in cited_eids if eid in state.evidence}
        if cited_ev:
            p3.append("EVIDENCE (cited by live hypotheses)")
            for eid in sorted(cited_ev):
                e = cited_ev[eid]
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
                p3.append(f"- {s(eid)} q={s(e['query_id'])} rows={rows}{f}: {e['claim']}")

    # ── P4: refuted hypotheses + orphaned/uncited evidence ────────────────────
    p4: list[str] = []
    if "hypotheses" in sections and refuted_h_lines:
        p4.append("HYPOTHESES (refuted)")
        p4.extend(refuted_h_lines)

    if "evidence" in sections and state.evidence:
        live_hids = {
            hid for hid, h in state.hypotheses.items() if h["status"] in ("live", "accepted")
        }
        cited_eids: set[str] = set()  # type: ignore[assignment]
        for hid in live_hids:
            h = state.hypotheses[hid]
            for e in h["evidence"]:
                if e["withdrawn"] is None:
                    cited_eids.add(e["evidence_id"])

        orphaned_ev = {
            eid: e for eid, e in state.evidence.items() if eid not in cited_eids
        }
        if orphaned_ev:
            p4.append("EVIDENCE (orphaned / cited by refuted hypotheses)")
            for eid in sorted(orphaned_ev):
                e = orphaned_ev[eid]
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
                p4.append(f"- {s(eid)} q={s(e['query_id'])} rows={rows}{f}: {e['claim']}")

    # ── Assemble with budget ──────────────────────────────────────────────────
    # P1 is highest priority but the hard ceiling still applies.
    result = _join(p1)

    buckets = [
        (p2, "unresolved failures and recent resolved objections"),
        (p3, "evidence cited by live hypotheses"),
        (p4, "refuted hypotheses and orphaned evidence"),
    ]
    omitted: list[str] = []
    for lines, label in buckets:
        if not lines:
            continue
        block = "\n" + _join(lines)
        if len(result) + len(block) <= max_chars:
            result += block
        else:
            remaining = max_chars - len(result)
            if remaining > 120:
                # partial fit: take as many complete lines as possible
                partial = block[:remaining].rsplit("\n", 1)[0]
                dropped = block[len(partial):].count("\n")
                result += partial
                omitted.append(f"{dropped} more lines of {label}")
            else:
                omitted.append(f"{len(lines)} more lines of {label}")

    if omitted:
        hint = "; ".join(omitted)
        result += (
            f"\n[... {hint} omitted to fit context window. "
            "Use ledger_view(sections=[...]) to retrieve them.]"
        )

    # Hard ceiling: if P1 itself exceeded max_chars, truncate and add hint.
    if len(result) > max_chars:
        cut = result[:max_chars].rsplit("\n", 1)[0]
        hidden = result[len(cut):].count("\n")
        result = cut + f"\n[... {hidden} more lines; call ledger_view with a section filter]"

    return result
