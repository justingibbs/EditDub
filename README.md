# EditDub

## Anthropic Haystack version

1. Create a virtual environment to run
```
python3 -m venv env_pinecone
source env_pinecone/bin/activate
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Create a .env file with 
```
# This is a comment, ignored by the program
ANTHROPIC_API_KEY=your_api_key_here
```

4. Add RAG files to `load_files` directory

These can be `.txt`, `.md`, `.pdf`.

5. Run program
```
streamlit run haystack_chat.py
```

```
python rag_chatbot.py
```

## Older version of program

1. Create a virtual environment to run
```
python3 -m venv venv
source venv/bin/activate
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Create a .env file with 
```
# This is a comment, ignored by the program
OPENAI_API_KEY=your_actual_openai_key_here
```

Install