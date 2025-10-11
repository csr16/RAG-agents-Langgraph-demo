# Langgraph RAG Agent with Ollama and Llama3.2

This project implements a Retrieval-Augmented Generation (RAG) agent using Langgraph, Ollama, and Llama3.2. The agent can perform document retrieval, web searches, and generate answers based on the retrieved information.

## Prerequisites

- Python 3.8+
- Ollama (for running Llama3.2 locally)
- Pinecone (for vector storage and retrieval)
- Tavily API key (for web search functionality)

## Installation (on Linux)

1. Clone this repository:

   ```
   git clone git@github.com:csr16/RAG-agents-Langgraph-demo.git
   cd RAG-agents-Langgraph-demo
   ```

2. Create a virtual environment and activate it:

   ```
   conda create -n your_env_name python=3.10
   conda activate your_env_name
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
   conda activate your_env_name

   which ollama
   ollama -v
   ```

6. Pull the Llama3.2 model:

   ```
   ollama pull llama3.2:3b
   ```

7. If you met following problem
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

   - `PINECONE_API_KEY`: Your Pinecone API key
   - `OLLAMA_MODEL`: llama3.2:3b
   - `PINECONE_INDEX_NAME`: Name of your Pinecone index
   - `COHERE_API_KEY`: Your Cohere API key for embeddings
   - `TAVILY_API_KEY`: Your Tavily API key for web search functionality
   - `EMBEDDING_MODEL`: embed-multilingual-v3.0
   - `RETRIEVER_K`: set the amount of docs u want to retrieve per query
   - `WEB_SEARCH_K`: set the amount of web searches for one query

## Usage

To run the Langgraph RAG agent, execute the following command:

``` 
python main.py 
```

The agent will prompt you to enter a question and a namespace for the Pinecone index.

## Project Structure

- `main.py`: Entry point of the application
- `agent.py`: Implements the Langgraph agent logic
- `utils/`:
  - `config.py`: Configuration management
  - `llm.py`: Ollama LLM integration
  - `retriever.py`: Document retrieval using Pinecone
  - `tools.py`: Web search tool implementation
  - `state.py`: State management for Langgraph

## Key Components

- **Retriever**: Uses Pinecone for vector storage and retrieval, with Cohere embeddings.
- **Web Search**: Utilizes Tavily for web search functionality.
- **LLM**: Integrates Ollama to run Llama3.2 locally for answer generation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.