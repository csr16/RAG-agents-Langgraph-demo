import logging
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from utils.llm import OllamaLLM
from utils.retriever import create_retriever
from utils.state import GraphState
from utils.config import Config
from langchain.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()


def create_agent():
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Define entry
    workflow.set_entry_point("retrieve")

    # Define edges and logic
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def retrieve(state: GraphState) -> GraphState:
    try:
        logger.info("Starting retrieval process")
        query = (state.get("question") or "").strip()
        if not query:
            state["error"] = "Empty question"
            state["current_step"] = "retrieve"
            return state

        retriever = state.get("retriever") or create_retriever()

        docs = retriever.invoke(query)
        if not docs and hasattr(retriever, "base_retriever"):
            logger.info("Compression yielded 0 docs; falling back to base retriever.")
            docs = retriever.base_retriever.invoke(query)

        ctx = state.get("context") or []
        ctx.extend([d.page_content for d in docs])

        wsr = state.get("web_search_results") or []
        if wsr:
            ctx.extend([str(x) for x in wsr])

        state["context"] = ctx
        state["current_step"] = "retrieve"
        logger.info("Retrieved %d documents", len(docs))
        return state

    except Exception as e:
        logger.exception("Error in retrieve")
        state["error"] = str(e)
        return state  

def generate_answer(state: GraphState) -> GraphState:
    try:
        logger.info("Starting answer generation process")
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        # Convert all context items to strings
        context_strings = []
        for item in state["context"]:
            if isinstance(item, dict):
                # If the item is a dictionary, convert it to a string representation
                context_strings.append(str(item))
            elif isinstance(item, str):
                context_strings.append(item)
            else:
                # For any other type, convert to string
                context_strings.append(str(item))

        prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use seven sentences maximum and keep the answer concise.

Answer:"""
        )

        chain = prompt | llm
        response = chain.invoke({
            "question": state["question"],
            "context": "\n".join(context_strings)
        })

        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state
    except Exception as e:
        logger.error(f"Error in generate_answer function: {str(e)}")
        state["error"] = str(e)
    finally:
        return state
