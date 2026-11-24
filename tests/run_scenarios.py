"""Run every conversation scenario against the real assistant and report pass/fail.
Usage (from the project folder):   python -m tests.run_scenarios        (each scenario once)
                                    python -m tests.run_scenarios 3      (each 3 times: finds flaky ones)"""
import re
import sys
import time

import pandas as pd

from src.rag.knowledge_base import detect_lang
from tests.scenarios import SCENARIOS


def failed_checks(expect: dict, out: dict) -> list[str]:
    """Compare the last reply with the expectations. Returns what went wrong (empty = pass)."""
    reply = re.sub(r"[\u00a0\u2009\u202f]", " ", out["reply"]).lower()   # special spaces -> normal space
    problems = []
    if "route" in expect and out["route"] != expect["route"]:
        problems.append(f"route={out['route']} (expected {expect['route']})")
    if "sources" in expect and bool(out["sources"]) != expect["sources"]:
        problems.append("expected sources" if expect["sources"] else f"unexpected sources {out['sources']}")
    if "source_any" in expect and not set(expect["source_any"]) & set(out["source_ids"]):
        problems.append(f"sources {out['source_ids']} (expected one of {expect['source_any']})")
    for group in expect.get("contains", []):
        if not any(word.lower() in reply for word in group):
            problems.append(f"reply missing one of {group}")
    for word in expect.get("not_contains", []):
        if word.lower() in reply:
            problems.append(f"reply contains {word!r}")
    if "lang" in expect and detect_lang(out["reply"]) != expect["lang"]:
        problems.append(f"reply language={detect_lang(out['reply'])}")
    if "ticket" in expect and bool(out["ticket"]) != expect["ticket"]:
        problems.append("expected a ticket" if expect["ticket"] else "unexpected ticket")
    for flag in ("upset", "injection"):
        if flag in expect and out[flag] != expect[flag]:
            problems.append(f"{flag}={out[flag]}")
    if "masked" in expect and ("[hidden" in out["masked_message"]) != expect["masked"]:
        problems.append("secret was not hidden")
    return problems


def play(assistant, turns: list[str]) -> dict:
    """Play a whole conversation; return the last reply (history kept like a real chat)."""
    history, out = [], None
    for message in turns:
        out = assistant.reply(message, history)
        history += [{"role": "user", "content": out["masked_message"]},
                    {"role": "assistant", "content": out["reply"]}]
    return out


def run(assistant, scenarios=SCENARIOS, repeats: int = 1) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for attempt in range(1, repeats + 1):
            start = time.perf_counter()
            try:
                out = play(assistant, scenario["turns"])
                problems = failed_checks(scenario["expect"], out)
                reply = out["reply"]
            except Exception as error:                       # a crash is a failure too
                problems, reply = [f"crashed: {error}"], ""
            rows.append({"scenario": scenario["name"], "attempt": attempt,
                         "passed": not problems, "problems": "; ".join(problems),
                         "seconds": round(time.perf_counter() - start, 1), "reply": reply})
            print(("PASS " if not problems else "FAIL ") + scenario["name"],
                  "" if not problems else "-> " + "; ".join(problems), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.factory import build_assistant
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    assistant = build_assistant(tickets_file="data/tickets_test.jsonl")   # test tickets kept apart
    results = run(assistant, repeats=repeats)
    results.to_csv("results/scenario_results.csv", index=False)
    rate = results.groupby("scenario")["passed"].mean()
    print(f"\n{int(results['passed'].sum())}/{len(results)} runs passed")
    flaky = rate[(rate > 0) & (rate < 1)]
    if len(flaky):
        print("Flaky (sometimes pass, sometimes fail):", ", ".join(flaky.index))
