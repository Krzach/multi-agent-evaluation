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
