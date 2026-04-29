from typing import Any, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def retrieve_node(state: dict, retriever: Any = None) -> dict:
    """
    Retrieves context based on the user's query.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"context": ""}

    # Get the latest user query
    query = messages[-1].content

    # If a real retriever is provided, you would call it here:
    if retriever:
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        # Fallback mock context
        context = f"Mock retrieved document for query: '{query}'. " \
                  f"The capital of France is Paris. Gaia is an ancient Greek goddess."

    return {"context": context}

def generate_node(state: dict, llm: Any) -> dict:
    """
    Takes the retrieved context and the user query to generate a response.
    """
    messages = state.get("messages", [])
    context = state.get("context", "")

    if not messages:
        return {"messages": []}

    query = messages[-1].content

    system_prompt = (
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know.\n\n"
        f"Context:\n{context}"
    )

    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    # Generate the response
    response = llm.invoke(prompt_messages)

    # Since our state uses operator.add for messages, we only need to return the new message
    return {"messages": [response]}