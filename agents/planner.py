"""
agents/planner.py — the Planner agent, STANDALONE VERSION.

Runs entirely on its own: reads obligations from a local JSON file (or an
in-memory list), asks the local model to draft a phased project plan, and
validates that plan with plain code (not the model) per the brief: every
deliverable covered, every payment scheduled before it, no circular
dependencies.

Swap in real depednencies and shared-memory calls later when ready.

Run: python agents/planner.py
"""

import json
import uuid
from pathlib import Path

MODEL_NAME = "llama3.1:8b"
MAX_RETRIES = 3

# Obligation types that MUST be covered by at least one plan item.
COVERAGE_REQUIRED_TYPES = ("deliverable", "milestone")

# 1. VALIDATION CHECKS — plain code, not the model, per the brief

def check_full_coverage(obligations: list[dict], plan_items: list[dict]) -> list[str]:
    """
    Returns a list of obligation ids that are NOT covered by any plan item's
    linked_obligation_ids, restricted to types that require coverage
    (deliverables and milestones — not raw assumptions).
    Empty list == check passes.
    """
    covered_ids = set()
    for item in plan_items:
        covered_ids.update(item["linked_obligation_ids"])

    uncovered = [
        obl["id"] for obl in obligations
        if obl["type"] in COVERAGE_REQUIRED_TYPES and obl["id"] not in covered_ids
    ]
    return uncovered


def check_payment_ordering(obligations: list[dict], plan_items: list[dict]) -> list[str]:
    """
    Returns a list of payment obligation ids that do NOT have any plan item
    whose scheduled_before field points to them (i.e. no work is explicitly
    scheduled ahead of that payment).
    Empty list == check passes.
    """
    scheduled_before_ids = {item["scheduled_before"] for item in plan_items if item.get("scheduled_before")}

    payment_ids = [obl["id"] for obl in obligations if obl["type"] == "payment"]
    unscheduled_payments = [pid for pid in payment_ids if pid not in scheduled_before_ids]
    return unscheduled_payments


def check_no_cycles(plan_items: list[dict]) -> list[str]:
    """
    Detects circular dependencies in plan_items' depends_on graph using DFS.
    Returns a list of node keys involved in a cycle (empty == check passes).

    Works on two shapes of plan_items:
      - Freshly-drafted (pre-write) items, keyed by "work_package" name —
        this is what depends_on references at draft time, since the model
        doesn't know database ids yet.
      - Already-written (post-write) items, keyed by "id" — used if this
        is ever called after memory.add_plan_item().
    Falls back to "work_package" when "id" isn't present.
    """
    def node_key(item):
        return item.get("id") or item["work_package"]

    graph = {node_key(item): item.get("depends_on", []) for item in plan_items}
    visited = set()
    in_progress = set()
    cycle_nodes = set()

    def dfs(node, path):
        if node in in_progress:
            cycle_start = path.index(node)
            cycle_nodes.update(path[cycle_start:])
            return
        if node in visited:
            return
        in_progress.add(node)
        path.append(node)
        for dep in graph.get(node, []):
            if dep in graph:
                dfs(dep, path)
        path.pop()
        in_progress.discard(node)
        visited.add(node)

    for plan_item_id in graph:
        if plan_item_id not in visited:
            dfs(plan_item_id, [])

    return list(cycle_nodes)


def validate_plan(obligations: list[dict], plan_items: list[dict]) -> dict:
    """
    Runs all three checks and returns a summary dict:
      { "passed": bool, "uncovered": [...], "unscheduled_payments": [...], "cycles": [...] }
    """
    uncovered = check_full_coverage(obligations, plan_items)
    unscheduled_payments = check_payment_ordering(obligations, plan_items)
    cycles = check_no_cycles(plan_items)

    return {
        "passed": not (uncovered or unscheduled_payments or cycles),
        "uncovered": uncovered,
        "unscheduled_payments": unscheduled_payments,
        "cycles": cycles,
    }

# 2. MODEL CALL — lazy imports, retries with validation feedback

def _build_prompt(obligations: list[dict], validation_feedback: dict = None) -> str:
    obligations_json = json.dumps(
        [{"id": o["id"], "type": o["type"], "description": o["description"]} for o in obligations],
        indent=2,
    )

    feedback_text = ""
    if validation_feedback:
        feedback_text = f"""
Your previous plan FAILED validation with these problems:
- Obligations not covered by any work package: {validation_feedback['uncovered']}
- Payments with no work scheduled before them: {validation_feedback['unscheduled_payments']}
- Plan items involved in a circular dependency: {validation_feedback['cycles']}
Fix ALL of these in your next plan.
"""

    return f"""You are drafting a project plan from a list of contract obligations.

Obligations:
{obligations_json}
{feedback_text}
Return a JSON array of work packages. Each work package must have:
- "phase": a phase name, e.g. "Phase 1: Discovery"
- "work_package": a short task name
- "linked_obligation_ids": array of obligation ids (from the list above) this work package covers
- "depends_on": array of OTHER work package names in THIS plan that must finish first (use the "work_package" name as the reference), or [] if none
- "scheduled_before": the obligation id of a PAYMENT obligation this work package must be completed before, or null if not applicable

Every deliverable and milestone obligation must be covered by at least one work package.
Every payment obligation must have at least one work package scheduled before it.
Do not create circular dependencies between work packages.

Return ONLY the JSON array, no other text.
"""


def call_model(obligations: list[dict], validation_feedback: dict = None, model: str = MODEL_NAME) -> list[dict]:
    import ollama
    from pydantic import BaseModel, ValidationError
    from typing import Optional, List

    class PlanItemDraft(BaseModel):
        phase: str
        work_package: str
        linked_obligation_ids: List[str]
        depends_on: List[str] = []
        scheduled_before: Optional[str] = None

    prompt = _build_prompt(obligations, validation_feedback)
    last_error = None

    for attempt in range(MAX_RETRIES):
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt if attempt == 0 else
                       f"{prompt}\n\nYour previous response was invalid: {last_error}"}],
            format="json",
        )
        raw = response["message"]["content"]

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array")
            validated = [PlanItemDraft(**item).model_dump() for item in parsed]
            return validated
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = str(e)
            continue

    return []


# 3. ENTRY POINT

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def load_obligations(path: str) -> list[dict]:
    """
    Loads obligations from a local JSON file. Expected shape — a list of:
      {"id": "OBL-1", "type": "deliverable", "description": "...",
       "amount": null, "due_date": null, "confidence": "high"}
    In the real pipeline this would come from memory.get_approved_obligations();
    for now it's a stand-in so the Planner can be built and tested without
    the Extractor or memory.py existing yet.
    """
    return json.loads(Path(path).read_text())


def run_planner(obligations: list[dict], max_model_retries: int = 2) -> dict:
    """
    Drafts a plan for the given obligations, validates it with code, and
    retries the MODEL (not just JSON parsing) if validation fails — feeding
    back exactly what was wrong.

    Returns:
      {
        "plan_items": [...],       # the final plan, with generated ids
        "validation": {...},        # result of the last validation check
      }
    """
    validation_feedback = None
    plan_draft = []
    result = {"passed": False, "uncovered": [], "unscheduled_payments": [], "cycles": []}

    for attempt in range(max_model_retries + 1):
        plan_draft = call_model(obligations, validation_feedback)
        result = validate_plan(obligations, plan_draft)
        if result["passed"]:
            break
        validation_feedback = result

    plan_items = []
    for item in plan_draft:
        plan_items.append({
            "id": _new_id("WP"),
            "phase": item["phase"],
            "work_package": item["work_package"],
            "linked_obligation_ids": item["linked_obligation_ids"],
            "depends_on": item.get("depends_on", []),
            "scheduled_before": item.get("scheduled_before"),
        })

    return {"plan_items": plan_items, "validation": result}


def save_plan(result: dict, out_path: str = "plan_output.json"):
    """Writes the planner's output to a local JSON file so it's inspectable
    without any database."""
    Path(out_path).write_text(json.dumps(result, indent=2))
    return out_path


if __name__ == "__main__":
    obligations = [
        {"id": "OBL-1", "type": "deliverable", "description": "Draft report"},
        {"id": "OBL-2", "type": "milestone", "description": "Kickoff meeting"},
        {"id": "OBL-3", "type": "payment", "description": "Final payment"},
    ]

    good_plan = [
        {"id": "WP-1", "phase": "Phase 1", "work_package": "Kickoff", "linked_obligation_ids": ["OBL-2"], "depends_on": []},
        {"id": "WP-2", "phase": "Phase 1", "work_package": "Write report", "linked_obligation_ids": ["OBL-1"], "depends_on": ["WP-1"], "scheduled_before": "OBL-3"},
    ]
    result = validate_plan(obligations, good_plan)
    assert result["passed"], f"Expected good plan to pass, got {result}"
    print("Good plan passed validation:", result)

    bad_plan = [
        {"id": "WP-1", "phase": "Phase 1", "work_package": "Kickoff", "linked_obligation_ids": ["OBL-2"], "depends_on": ["WP-2"]},
        {"id": "WP-2", "phase": "Phase 1", "work_package": "Something else", "linked_obligation_ids": [], "depends_on": ["WP-1"]},
    ]
    result = validate_plan(obligations, bad_plan)
    assert not result["passed"]
    assert "OBL-1" in result["uncovered"]
    assert "OBL-3" in result["unscheduled_payments"]
    assert len(result["cycles"]) == 2
    print("Bad plan correctly failed validation:", result)

    print("\nAll validation smoke tests passed.")

    from unittest.mock import patch

    def fake_model_output(obligations, validation_feedback=None, model=MODEL_NAME):
        deliverable = next(o for o in obligations if o["type"] == "deliverable")
        milestone = next(o for o in obligations if o["type"] == "milestone")
        payment = next(o for o in obligations if o["type"] == "payment")
        return [
            {"phase": "Phase 1", "work_package": "Kickoff", "linked_obligation_ids": [milestone["id"]], "depends_on": []},
            {"phase": "Phase 1", "work_package": "Write report", "linked_obligation_ids": [deliverable["id"]],
             "depends_on": ["Kickoff"], "scheduled_before": payment["id"]},
        ]

    with patch("__main__.call_model", side_effect=fake_model_output):
        run_result = run_planner(obligations)

    assert run_result["validation"]["passed"], f"Expected standalone run to pass, got {run_result['validation']}"
    assert len(run_result["plan_items"]) == 2
    out_path = save_plan(run_result, out_path="test_plan_output.json")
    print(f"\nStandalone run_planner() test passed. Output written to {out_path}:")
    print(json.dumps(run_result, indent=2))
    #Path(out_path).unlink()

    print("\nAll planner smoke tests passed.")