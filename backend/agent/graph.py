from langgraph.graph import StateGraph, END
from backend.agent.state import SimulationState
from backend.agent.nodes import planner_node, executor_node, analyst_node, should_retry

def create_simulation_graph():
    workflow = StateGraph(SimulationState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("analyst", analyst_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_retry,
        {
            "retry": "planner",
            "proceed": "analyst"
        }
    )
    
    workflow.add_edge("analyst", END)

    return workflow.compile()

simulation_app = create_simulation_graph()