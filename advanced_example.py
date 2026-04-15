"""Advanced LangChain example with agents and tools."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def main():
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Define tools
    tools = [multiply, add]

    # LangChain 1.x: graph-based agent (no LangChain Hub prompt required)
    agent = create_agent(
        llm,
        tools,
        system_prompt=(
            "You are a helpful assistant. Use the multiply and add tools for "
            "arithmetic. Report the final numeric result clearly."
        ),
    )

    # Run the agent (state is message-based)
    result = agent.invoke(
        {"messages": [HumanMessage(content="Calculate: 25 * 4 + 10")]}
    )

    last = result["messages"][-1]
    output = last.content if isinstance(last, AIMessage) else str(last)

    print(f"\nFinal Result: {output}")


if __name__ == "__main__":
    main()
