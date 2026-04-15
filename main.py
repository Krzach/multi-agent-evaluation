"""Minimal LangChain working example."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def main():
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-5.4", temperature=0)

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("user", "{input}")]
    )

    # Create an output parser
    output_parser = StrOutputParser()

    # Chain components together
    chain = prompt_template | llm | output_parser

    # Run the chain
    response = chain.invoke({"input": "What is 2 + 2?"})
    print(f"Question: What is 2 + 2?")
    print(f"Answer: {response}")

    # Another example with more complex input
    response2 = chain.invoke({"input": "Explain quantum computing in one sentence."})
    print(f"\nQuestion: Explain quantum computing in one sentence.")
    print(f"Answer: {response2}")


if __name__ == "__main__":
    main()
