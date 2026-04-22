Overview of RAG Chat which involves two agents, including a RAG User Proxy and a Retrieval-augmented Assistant.

To use Retrievalaugmented Chat, one needs to initialize two agents including Retrieval-augmented User Proxy and Retrieval-augmented Assistant.

## Retrieval-augmented User Proxy

- Given a set of documents, the Retrieval-augmented User Proxy
- Initializing the Retrieval-Augmented User Proxy necessitates specifying a path to the document collection.
- the RetrievalAugmented User Proxy can download the documents, segment them into chunks of a specific size, compute embeddings, and store them in a vector database.
- Once a chat is initiated, the agents collaboratively engage in code generation or question-answering
- retrieves relevant data from the DB
- executes the code if needed
- updates the context if needed

## Retrieval-augmented Assistant

- uses LLM to generate code or text to answer questions based on the question and context provided.
- If unable to produce a satisfactory response, replies with “Update Context” to the Retrieval-Augmented User Proxy.

## Detailed Workflow.

1. User Inputs a question
2. For a given user input, the Retrieval-Augmented User Proxy retrieves document chunks based on the embedding similarity, and sends them along with the question to the Retrieval-Augmented Assistant. 4. Retrieval-augmented Assistant, which uses LLM to generate code or text to answer questions based on the question and context provided. If the LLM is unable to produce a satisfactory response, it is instructed to reply with “Update Context” to the Retrieval-Augmented User Proxy.
3. If a response includes code blocks, the Retrieval-Augmented User Proxy executes the code and sends the output as feedback.
4. If there are no code blocks or instructions to update the context, it terminates the conversation. Otherwise, it updates the context and forwards the question along with the new context to the Retrieval-Augmented Assistant. Note that if human input solicitation is enabled, individuals can proactively send any feedback, including “Update Context”, to the Retrieval-Augmented Assistant.
5. If the Retrieval-Augmented Assistant receives “Update Context”, it requests the next most similar chunks of documents as new context from the Retrieval-Augmented User Proxy. Otherwise, it generates new code or text based on the feedback and chat history. If the LLM fails to generate an answer, it replies with “Update Context” again.
6. This process can be repeated several times. Agents converse until they find a satisfactory answer. The conversation terminates if no more documents are available for the context.

We utilize Retrieval-Augmented Chat in two scenarios.

- The first scenario aids in generating code based on a given codebase. While LLMs possess strong coding abilities, they are unable to utilize packages or APIs that are not included in their training data, e.g., private codebases, or have trouble using trained ones that are frequently updated post-training. Hence, Retrieval-Augmented Code Generation is considered to be highly valuable.
- The second scenario involves question-answering on the GAIA dataset
