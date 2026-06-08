from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import retriever_node, generator_node, critic_node

MAX_RETRIES = 1   # one retry is enough; same query = same chunks


def route_after_critic(state: AgentState):
    if not state.get("is_grounded") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retriever"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic",    critic_node)

    graph.set_entry_point("retriever")

    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retriever": "retriever", "end": END}
    )

    return graph.compile()


agent_graph = build_graph()
