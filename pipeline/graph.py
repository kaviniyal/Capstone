"""
LangGraph orchestration pipeline for insurance fraud analysis.

Flow:
  validate_input
      → retrieve_claims
          → [risk_scoring, policy_validation]  (parallel)
              → human_review_check  (HITL gate)
                  → recommendation
                      → END

HITL: if risk_scoring flags requires_human_review=True, the graph pauses
      at the 'awaiting_human_review' node and waits for a human decision
      via the /analyze/resume endpoint before continuing.
"""

import os
import sys
from typing import TypedDict, Annotated, Any
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from guardrails.guards import validate_and_sanitize
from agents.fraud_retrieval_agent import run_fraud_retrieval_agent
from agents.risk_scoring_agent import run_risk_scoring_agent
from agents.policy_validation_agent import run_policy_validation_agent
from agents.recommendation_agent import run_recommendation_agent


# ── State ───────────────────────────────────────────────────────────────────

class ClaimState(TypedDict):
    # inputs
    original_query: str
    filters: dict

    # guardrails
    sanitized_query: str
    guardrail_flags: list[str]

    # retrieval
    retrieved_claims: list[dict]
    query_used: str
    crag_triggered: bool

    # agent outputs
    risk_assessment: dict
    policy_validation: dict

    # hitl
    awaiting_human: bool
    human_decision: str   # "approve" | "escalate" | "reject" | ""

    # final
    recommendation: dict
    error: str


# ── Node implementations ─────────────────────────────────────────────────────

def node_validate_input(state: ClaimState) -> dict:
    result = validate_and_sanitize(state["original_query"])
    return {
        "sanitized_query": result["sanitized_text"],
        "guardrail_flags": result["flags"],
    }


def node_retrieve_claims(state: ClaimState) -> dict:
    if state.get("guardrail_flags") and "BLOCKED" in state["guardrail_flags"]:
        return {"retrieved_claims": [], "query_used": state["sanitized_query"], "crag_triggered": False}

    retrieval = run_fraud_retrieval_agent(
        query=state["sanitized_query"],
        top_k=5,
        filters=state.get("filters") or None,
    )
    return {
        "retrieved_claims": retrieval["claims"],
        "query_used":       retrieval["query_used"],
        "crag_triggered":   retrieval["crag_triggered"],
    }


def node_risk_scoring(state: ClaimState) -> dict:
    risk = run_risk_scoring_agent(
        query=state["sanitized_query"],
        retrieved_claims=state["retrieved_claims"],
    )
    return {"risk_assessment": risk}


def node_policy_validation(state: ClaimState) -> dict:
    validation = run_policy_validation_agent(
        query=state["sanitized_query"],
        retrieved_claims=state["retrieved_claims"],
    )
    return {"policy_validation": validation}


def node_hitl_check(state: ClaimState) -> dict:
    """Pause if the risk score falls in the human-review uncertainty band."""
    requires_review = state.get("risk_assessment", {}).get("requires_human_review", False)
    return {
        "awaiting_human": requires_review,
        "human_decision": state.get("human_decision", ""),
    }


def node_recommendation(state: ClaimState) -> dict:
    rec = run_recommendation_agent(
        query=state["sanitized_query"],
        risk_assessment=state["risk_assessment"],
        policy_validation=state["policy_validation"],
        retrieved_claims=state["retrieved_claims"],
    )
    return {"recommendation": rec}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_hitl(state: ClaimState) -> str:
    if state.get("awaiting_human") and not state.get("human_decision"):
        return "awaiting_human_review"   # pause — interrupt here
    return "recommendation"


def route_after_guardrail(state: ClaimState) -> str:
    if "BLOCKED" in (state.get("guardrail_flags") or []):
        return END
    return "retrieve_claims"


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(ClaimState)

    builder.add_node("validate_input",      node_validate_input)
    builder.add_node("retrieve_claims",     node_retrieve_claims)
    builder.add_node("risk_scoring",        node_risk_scoring)
    builder.add_node("policy_validation",   node_policy_validation)
    builder.add_node("hitl_check",          node_hitl_check)
    builder.add_node("awaiting_human_review", lambda s: s)  # pause node
    builder.add_node("recommendation",      node_recommendation)

    builder.set_entry_point("validate_input")

    builder.add_conditional_edges("validate_input", route_after_guardrail, {
        "retrieve_claims": "retrieve_claims",
        END: END,
    })

    # After retrieval, run risk scoring and policy validation in sequence
    # (LangGraph doesn't have native parallel fan-out without Send API;
    #  we run them back-to-back which is effectively sequential but clean)
    builder.add_edge("retrieve_claims",   "risk_scoring")
    builder.add_edge("risk_scoring",      "policy_validation")
    builder.add_edge("policy_validation", "hitl_check")

    builder.add_conditional_edges("hitl_check", route_after_hitl, {
        "awaiting_human_review": "awaiting_human_review",
        "recommendation":        "recommendation",
    })

    # After human resumes (human_decision set), go straight to recommendation
    builder.add_edge("awaiting_human_review", "recommendation")
    builder.add_edge("recommendation", END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["awaiting_human_review"],
    )


# Compiled graph singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_analysis(query: str, filters: dict | None = None, thread_id: str = "default") -> dict:
    """
    Run the full pipeline. Returns the final state dict.
    If HITL is triggered, state['awaiting_human'] will be True
    and the caller must call resume_analysis() after human decision.
    """
    graph = get_graph()
    initial_state: ClaimState = {
        "original_query":   query,
        "filters":          filters or {},
        "sanitized_query":  "",
        "guardrail_flags":  [],
        "retrieved_claims": [],
        "query_used":       "",
        "crag_triggered":   False,
        "risk_assessment":  {},
        "policy_validation":{},
        "awaiting_human":   False,
        "human_decision":   "",
        "recommendation":   {},
        "error":            "",
    }
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config=config)
    return dict(final_state)


def resume_analysis(thread_id: str, human_decision: str) -> dict:
    """
    Resume a paused (HITL) pipeline with the human investigator's decision.
    human_decision: "approve" | "escalate" | "reject"
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"human_decision": human_decision, "awaiting_human": False})
    final_state = graph.invoke(None, config=config)
    return dict(final_state)
