# deeprage
a multi-model RAGE MVP for use with local and external models from API

```txt
RAGE/
├── src/
│   ├── __init__.py
│   ├── memory.py          # Memory management (LTM, STM, context)
│   ├── models.py          # AI model handlers (Ollama, Groq, etc.)
│   ├── openmind.py        # Central configuration and resource management
├── data/
│   ├── knowledge/         # Directory for long-term memory (LTM)
│   └── conversations/     # Directory for short-term memory (STM)
├── logs/
│   └── rage.log           # Log file for debugging and monitoring
├── .env                   # Environment variables
├── requirements.txt       # Dependencies
├── styles.css             # CSS styling for the UI
├── prompt.txt             # System prompt
└── rage.py                # Main Streamlit UI application
```
