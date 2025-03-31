# Oil Well Diagnostic RAG System

A Rasa-based conversational AI assistant for oil well diagnostics and monitoring using Retrieval Augmented Generation (RAG).

## Features

- Interactive chatbot interface for querying oil well status and diagnostics
- Retrieval-augmented generation using Hugging Face models
- Vector database using Chroma for semantic search
- Custom actions for processing well status queries
- Support for sensor-specific queries with context-aware responses

## Technologies

- Rasa: Conversational AI framework
- LangChain: For building RAG pipelines
- HuggingFace: For language models and embeddings
- Chroma: Vector database for semantic search
- PyTorch: For machine learning models

## Installation

1. Clone the repository
2. Set up a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start the Rasa server:
```bash
rasa run --enable-api
```
5. In a separate terminal, start the action server:
```bash
rasa run actions
```

## Usage

You can interact with the assistant by sending queries like:
- "What is the status of well 1?"
- "Tell me about the T-PDG sensor"
- "How does oil extraction work?"

## Project Structure

- `actions/`: Contains custom actions including RAG implementation
- `data/`: Training data for NLU and stories
- `models/`: Trained Rasa models
- `chroma_db/`: Vector store for document embeddings 