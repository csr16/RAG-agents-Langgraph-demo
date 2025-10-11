# A Minimal Demo for Constructing Langgraph RAG Agent with Ollama and Llama3.2

A Minimal Demo: Building a LangGraph RAG Agent with Ollama & Llama 3.2

This repository demonstrates how to construct a **Retrieval-Augmented Generation (RAG)** agent using **LangGraph**, **Ollama**, and **Llama 3.2**.
The agent performs document retrieval and generation following the same pipeline used in **MP1**, using **Pyserini** for retrieval and **LangGraph** for orchestration.

## Features

- **LangGraph-based RAG pipeline** – clean, modular agent design.

- **Local LLM inference via Ollama** – no API keys required.

- **Pyserini BM25 retriever** – replicates classical IR methods.

- **Configurable hyperparameters** via `.env`.

- **Fully offline workflow** for reproducible experiments.

## Prerequisites

- **Python** ≥ 3.10
- **Conda** (recommended)
- **Ollama** (for running Llama3.2 locally)

## Installation (Linux)

1. Clone this repository:

   ```
   git clone git@github.com:csr16/RAG-agents-Langgraph-demo.git
   cd RAG-agents-Langgraph-demo
   ```

2. Create a virtual environment and activate it:

   ```
   conda create -n rag-demo python=3.10 -y
   conda activate rag-demo
   ```

3. Install Pyserini:
   ```
   # Inside the new environment...
   conda install -c conda-forge openjdk=21 maven -y

   # from https://pytorch.org/get-started/locally/
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # If you want the optional dependencies, otherwise skip
   conda install -c pytorch faiss-cpu -y

   # Good idea to always explicitly specify the latest version, found here: https://pypi.org/project/pyserini/
   pip install pyserini
   ```

4. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

5. Install Ollama:

   ```
   mkdir -p ~/opt/ollama ~/bin # Or anywhere you want to set as root dir for ollama
   curl -L "https://ollama.com/download/ollama-linux-amd64.tgz" -o /tmp/ollama.tgz
   tar -C ~/opt/ollama -xzf /tmp/ollama.tgz

   echo 'export PATH="$HOME/opt/ollama/bin:$PATH"' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH="$HOME/opt/ollama/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
   echo 'export OLLAMA_MODELS="$HOME/data/ollama_models"' >> ~/.bashrc
   mkdir -p ~/data/ollama_models

   source ~/.bashrc
   ```

6. Pull the Llama3.2 model:

   ```
   ollama pull llama3.2:3b
   ```

7. TroubleShooting

   If you encounter the following error:
   ```
   numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)
   ```

   try:
   ```
   conda remove -n your_env_name -y numpy
   conda clean -a -y
   conda install -n your_env_name -y -c conda-forge numpy=1.26.4 scipy=1.11.4 scikit-learn=1.3.2
   pip uninstall packaging
   pip install packaging==23.2
   ```

## Configuration

1. Create a `.env` file in the project root directory.

2. Set the following environment variables in the `.env` file:

   - `PYSERINI_CNAME`: The dataset we want to use to build indexes and retrieve documents
   - `OLLAMA_MODEL`: llama3.2:3b
   - `RETRIEVER_K`: set the amount of docs u want to retrieve per query
   - `PYSERINI_K1`: The hyperparameter of bm25
   - `PYSERINI_B`: The hyperparameter of bm25

## Usage

Ensure your dataset is placed under the `data/` directory, then run::

``` 
python main.py 
```

The agent will prompt you to enter a question and a namespace for the Pinecone index.

## Project Structure

- `data/`: Local document corpus (e.g., MP1 docs)
- `main.py`: Entry point of the application
- `agent.py`: Implements the Langgraph agent logic
- `utils/`:
  - `config.py`: Configuration management
  - `llm.py`: Ollama LLM integration
  - `retriever.py`: Document retrieval using Pyserini
  - `state.py`: State management for Langgraph

## Key Components

- **Retriever**: Uses Pyserini for buiding indexes and retriever.
- **LLM**: Integrates Ollama to run Llama3.2 locally for answer generation.
- **Web Search (Optional, Not implemented yet)** – Integrates with Tavily for web-based retrieval.
