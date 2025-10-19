# Legal RAG Chatbot

A multi-turn conversation RAG (Retrieval-Augmented Generation) chatbot specialized in legal questions, particularly human rights law.

## Features

- 📚 **Document Processing**: Processes PDF documents and creates searchable chunks
- 🔍 **Semantic Search**: Uses all-MiniLM-L6-v2 embeddings with FAISS for efficient retrieval
- 💬 **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- 🎯 **Session Management**: Separate chat sessions with individual histories
- ⚡ **Fast Generation**: Powered by Groq's LLaMA models
- 🌐 **Web Interface**: User-friendly Streamlit interface

## Setup

1. **Create Virtual Environment**:
   ```bash
   cd legal_chatbot
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   - Create a `.env` file in the `legal_chatbot` folder
   - Add your Groq API key to `.env`:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

5. **Add Documents**:
   - Place your PDF documents in the `data/` folder

6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Initialize System**: Click "Initialize System" in the sidebar to process documents
2. **Create Session**: Start a new chat session
3. **Ask Questions**: Type legal questions and get context-aware responses
4. **Manage Sessions**: Switch between different conversation sessions

## Virtual Environment Benefits

- ✅ Isolated dependencies
- ✅ No conflicts with system Python
- ✅ Reproducible environment
- ✅ Easy dependency management

## Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

## File Structure

```
legal_chatbot/
├── venv/                    # Virtual environment (created after setup)
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── conversation_manager.py
│   └── rag_pipeline.py
├── data/
│   ├── README.md
│   └── (your PDFs here)
├── vector_store/           # Generated vector indices
├── app.py
├── run_with_venv.py       # Helper script to run with venv
├── requirements.txt
├── .env                   # API keys
└── README.md
```

## Troubleshooting

- **Import Errors**: Make sure virtual environment is activated
- **Missing Dependencies**: Run `pip install -r requirements.txt` in activated venv
- **API Errors**: Check if GROQ_API_KEY is set correctly in `.env`
- **No Documents**: Add PDFs to `data/` folder and reinitialize system
