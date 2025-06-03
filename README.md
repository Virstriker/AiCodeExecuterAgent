# AI Code Executer

A terminal-based AI chatbot powered by Google's Gemini API that can execute Python code and show results.

## Features

- Interactive terminal-based chat interface
- Uses Google's Gemini AI for generating responses and code
- Automatically detects and executes Python code in responses
- Shows execution results to the user
- Colored output for better readability

## Requirements

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. Clone this repository or download the files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   
   Either set it as an environment variable:
   ```bash
   # For Windows PowerShell
   $env:GEMINI_API_KEY = "your-api-key-here"
   
   # For Linux/MacOS
   export GEMINI_API_KEY="your-api-key-here"
   ```
   
   Or enter it when prompted during execution.

## Usage

Run the chatbot:

```bash
python ai_code_executer.py
```

### Commands

- `exit`, `quit`, or `bye`: End the session
- `clear`: Clear the conversation history

## How It Works

1. The chatbot connects to the Gemini API and initializes a chat session.
2. The user types a question or request.
3. If the AI's response contains Python code (enclosed in ```python code ``` blocks), the code is automatically extracted and executed.
4. The execution result is shown to the user and sent back to the AI for further commentary.

## Security Note

The code execution happens in a separate process with a timeout limit to prevent harmful operations.
However, use caution when executing code from any AI model.

## License

MIT License
