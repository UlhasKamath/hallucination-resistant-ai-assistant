from typing import TypedDict, Optional


class AgentState(TypedDict):
    user_question:    str
    retrieved_chunks: Optional[str]
    answer:           Optional[str]
    is_grounded:      Optional[bool]
    retry_count:      int
    filter_source:    Optional[str]
