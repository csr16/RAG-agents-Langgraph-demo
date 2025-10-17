"""
agent.py
=========================
LangGraph-based RAG agent for MP2.

Pipeline:
1) retrieve(state)        -> collects context via Pyserini BM25 (+ optional compression)
2) generate_answer(state) -> prompts an LLM (Ollama) with the retrieved context

State keys (GraphState):
- question: str                     # user question (required)
- context: List[str]                # accumulated evidence passages
- web_search_results: Optional[Any] # optional external results (strings or dicts)
- retriever: Optional[BaseRetriever]# allow injection (for testing)
- final_answer: Optional[str]       # model's answer
- error: Optional[str]              # error message if any
- current_step: str                 # "retrieve" | "generate_answer"
"""

import logging
from typing import List

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

from utils.llm import OllamaLLM
from utils.retriever import create_retriever
from utils.state import GraphState
from utils.config import Config

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


# ---------------------------------------------------------------------
# Graph construction (entry for main.py)
# ---------------------------------------------------------------------
def create_agent():
    """
    Build and compile the LangGraph workflow:
       retrieve -> generate_answer -> END
    """
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Entry and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------
# Node 1: Retrieval
# ---------------------------------------------------------------------
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve evidence passages for the given question.

    Behavior:
    - Uses contextual compression retriever by default.
    - Falls back to base BM25 retriever if compression returns 0 docs.
    - Appends optional web_search_results into the context (if present).
    """
    try:
        logger.info("Starting retrieval")

        query = (state.get("question") or "").strip()
        if not query:
            state["error"] = "Empty question"
            state["current_step"] = "retrieve"
            logger.warning("Retrieval aborted: empty question")
            return state

        # Allow DI for tests; otherwise create from config
        retriever = state.get("retriever") or create_retriever()

        # Retrieve documents (ContextualCompressionRetriever or BaseRetriever)
        docs = retriever.invoke(query)
        if not docs and hasattr(retriever, "base_retriever"):
            logger.info("Compression yielded 0 docs; falling back to base retriever.")
            docs = retriever.base_retriever.invoke(query)

        # Build/extend context
        ctx: List[str] = list(state.get("context") or [])
        ctx.extend([getattr(d, "page_content", str(d)) for d in docs])

        # Optionally merge web search results
        wsr = state.get("web_search_results") or []
        if wsr:
            ctx.extend([str(x) for x in wsr])

        state["context"] = ctx
        state["current_step"] = "retrieve"
        logger.info(f"Retrieved {len(docs)} documents")
        return state

    except Exception as e:
        logger.exception("Error in retrieve")
        state["error"] = str(e)
        state["current_step"] = "retrieve"
        return state


# ---------------------------------------------------------------------
# Helpers for prompting
# ---------------------------------------------------------------------
def build_prompt(max_sentences: int = 7) -> ChatPromptTemplate:
    """
    Create the QA prompt. Keep it short and grounded in provided context.

    Tip for students:
    - You can experiment with the instruction style here (e.g., chain-of-thought
      vs. concise answers, citing sources, etc.).
    """
    template = f"""You are an assistant for question answering.

Use ONLY the context below to answer the question. If the answer is not in the context, say you don't know.

Context:
{{context}}

Question:
{{question}}

Answer in at most {max_sentences} sentences, concise and to the point.
Answer:"""
    return ChatPromptTemplate.from_template(template)


# ---------------------------------------------------------------------
# Node 2: Answer generation
# ---------------------------------------------------------------------
def generate_answer(state: GraphState) -> GraphState:
    """
    Generate a concise answer using Ollama with the retrieved context.
    Safely stringifies all context items before prompting.
    """
    try:
        logger.info("Starting answer generation")

        # Initialize local LLM (Ollama)
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        # Normalize context to strings (defensive for mixed types)
        context_strings: List[str] = []
        for item in state.get("context") or []:
            context_strings.append(item if isinstance(item, str) else str(item))

        prompt = build_prompt(max_sentences=7)
        chain = prompt | llm

        response = chain.invoke({
            "question": state.get("question", ""),
            "context": "\n".join(context_strings)
        })

        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state

    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        state["error"] = str(e)
        state["current_step"] = "generate_answer"
        return state
