from langgraph.graph import StateGraph, END
from backend.agent.state import SimulationState
from backend.agent.nodes import (
    scenario_parser_node,
    planner_node,
    critic_node,
    executor_node,
    validator_node,
    stats_node,
    analyst_node,
    should_critic_retry,
    should_retry,
)

def create_simulation_graph():
    """
    Universal AI Simulation Engine — Phase 1 graph (scientifically honest).

    Flow:
        scenario_parser
              ↓
           planner ◄──────────────────────────────┐
              ↓                                    │ rework (critic retry)
           critic ─────────────────────────────────┘
              │
            proceed
              ↓
           executor ◄──────────────────────────────┐
              ↓                                    │ retry (exec error retry)
         should_retry ──────────────────────────────┘
              │
            proceed
              ↓
          validator    (pure Python — no LLM — 10 consistency checks)
              ↓
            stats
              ↓
           analyst
              ↓
             END
    """
    workflow = StateGraph(SimulationState)

    # Register all nodes
    workflow.add_node("scenario_parser", scenario_parser_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("stats", stats_node)
    workflow.add_node("analyst", analyst_node)

    # Entry point
    workflow.set_entry_point("scenario_parser")

    # Linear edges
    workflow.add_edge("scenario_parser", "planner")
    workflow.add_edge("planner", "critic")
    workflow.add_edge("validator", "stats")
    workflow.add_edge("stats", "analyst")
    workflow.add_edge("analyst", END)

    # Conditional: Critic → rework (back to planner) OR proceed (to executor)
    workflow.add_conditional_edges(
        "critic",
        should_critic_retry,
        {
            "rework": "planner",
            "proceed": "executor",
        }
    )

    # Conditional: Executor → retry (back to planner) OR proceed (to validator)
    workflow.add_conditional_edges(
        "executor",
        should_retry,
        {
            "retry": "planner",
            "proceed": "validator",
        }
    )

    return workflow.compile()

simulation_app = create_simulation_graph()