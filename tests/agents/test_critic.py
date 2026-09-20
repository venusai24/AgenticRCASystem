"""critic acceptance (PLAN.md #7): fresh context, origin hidden, objections recorded, no failure-mode list."""
from agentic_rca.agents.critic import run_critic_round
from agentic_rca.llm import ScriptedLLM, call, load_prompt

AUDIT = ["tomcat", "mysql", "redis", "apache", "jvm", "garbage", "servicetest", "uocp", "k6", "dns",
         "certificate", "connection pool", "cpu", "memory", "disk", "kafka", "postgres", "kubernetes", "nginx"]


def test_round_is_fresh_hides_origin_and_records_objections(runtime):
    led = runtime.ledger
    led.apply("hypothesis_create_human", {"statement": "a colleague's hunch"}, actor="human")
    led.apply("hypothesis_create", {"statement": "lead idea"}, actor="lead")
    llm = ScriptedLLM(by_role={"critic": [
        call("objection_raise", text="no contrast cited", target={"hypothesis_id": "hy001"}),
        call("finish_round")]}).bind(led.con, led.run_id)
    res, raised = run_critic_round(ledger=led, runtime=runtime, llm=llm, round_no=1, tool_names=["sql"])
    assert res.status == "round_done" and len(raised) == 1
    first_context = llm.seen[0][1]
    assert len(first_context) == 2  # system + header only: no inherited trajectory
    assert "origin" not in first_context[1]["content"]
    assert led.state().objections[raised[0]]["round"] == 1
    # round 2 is a new loop that still sees round-1 objections via the ledger
    llm2 = ScriptedLLM(by_role={"critic": [call("finish_round")]}).bind(led.con, led.run_id)
    run_critic_round(ledger=led, runtime=runtime, llm=llm2, round_no=2, tool_names=["sql"])
    assert "no contrast cited" in llm2.seen[0][1][1]["content"] and len(llm2.seen[0][1]) == 2


def test_prompts_contain_no_domain_names():
    for name in ("critic_system", "lead_system", "step_summary"):
        text = load_prompt(name)[0].lower()
        assert not [w for w in AUDIT if w in text], name
