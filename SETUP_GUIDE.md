# LangChain Minimal Working Example

This project demonstrates basic LangChain usage with two examples: basic prompt chaining and agent-based interaction.

## Project Structure

- `main.py` - Basic LangChain example with prompt templates and chains
- `advanced_example.py` - Agent-based example with tools
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

## Running the Examples

### Basic Example (Prompt Chaining)

```bash
python main.py
```

This demonstrates:
- Initializing a language model (GPT-3.5-turbo)
- Creating a prompt template
- Building a chain (prompt → llm → parser)
- Invoking the chain with different inputs

### Advanced Example (Agents with Tools)

```bash
python advanced_example.py
```

This demonstrates:
- Defining custom tools (`multiply`, `add`)
- Creating a ReAct agent
- Agent executor with tool calling
- Complex multi-step reasoning

## Key LangChain Concepts

### 1. **LLMs**
The core language model, initialized with configuration (model name, temperature, API key).

### 2. **Prompts**
Templates that structure inputs to the LLM using `ChatPromptTemplate`.

### 3. **Output Parsers**
Convert raw LLM output into structured formats (strings, JSON, etc.).

### 4. **Chains**
Combine components using the pipe operator (`|`): `prompt | llm | parser`

### 5. **Tools**
Functions that agents can call to perform actions beyond text generation.

### 6. **Agents**
Intelligent systems that decide which tools to use based on user input.

## Troubleshooting

- **API Key Errors**: Ensure `.env` is in the working directory and `OPENAI_API_KEY` is set
- **Module Errors**: Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- **Rate Limiting**: OpenAI API has rate limits; space out requests if needed

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
