#!/usr/bin/env python3
"""
Gemini AI Code Executing Agent
A terminal-based chatbot using Google's Gemini API that can execute Python code and show results.
"""

import os
import sys
import re
import tempfile
import subprocess
import traceback
from io import StringIO
import google.generativeai as genai
from colorama import Fore, Style, init
from dotenv import load_dotenv
# Removed: import requests
# Removed: from bs4 import BeautifulSoup
# Removed: import json # Retained as it's generally useful
# Removed: import urllib.parse
# Removed: import time
import json # json is still useful for general purposes

# Attempt to import TavilyClient, will be handled gracefully if not found
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None # Placeholder if not installed

# Initialize colorama for cross-platform colored terminal output
init()

# Common standard library modules to exclude from pip install attempts
STANDARD_LIBRARY_MODULES = set([
    "os", "sys", "re", "tempfile", "subprocess", "traceback", "io", "json",
    "datetime", "collections", "math", "random", "argparse", "logging",
    "unittest", "itertools", "functools", "pathlib", "shutil", "time",
    "threading", "multiprocessing", "socket", "http", "urllib", "csv",
    "xml", "zipfile", "tarfile", "gzip", "bz2", "lzma", "hashlib", "hmac",
    "ssl", "asyncio", "concurrent", "contextlib", "dataclasses", "enum",
    "glob", "inspect", "pickle", "pprint", "queue", "signal", "statistics",
    "struct", "typing", "uuid", "warnings", "weakref", "webbrowser", "zoneinfo",
    "google", "colorama", "dotenv", # Also exclude project-used packages
    "tkinter", "tk", "Tkinter" # Common built-in GUI packages that can't be installed via pip
])

# Mapping of package names to their actual pip install names
PACKAGE_INSTALL_MAPPING = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "matplotlib.pyplot": "matplotlib",
    "yaml": "pyyaml",
}

class AiCodeExecuter:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.chat_session = None
        self.max_code_retries = 3
        self.max_dependency_install_retries = 3
        self.max_search_results = 10
        self.system_prompt = (
            "You are an AI coding assistant. Your primary role is to help with programming tasks by "
            "providing concise, executable Python code solutions.\n"
            "When a user's request involves code:\n"
            "1. Provide the Python code solution formatted within ```python and ``` tags for automatic execution.\n"
            "2. Briefly explain what the code does after providing it.\n\n"
            "File Processing:\n"
            "- Users can upload files for you to analyze or process.\n"
            "- When a file is uploaded, you'll receive it along with the user's instructions about what to do with it.\n"
            "- Respond based on the content of the file and the user's instructions.\n\n"
            "Web Search:\n"
            "- You can receive web search results from the user's search queries.\n"
            "- Analyze these search results to provide relevant information and code solutions.\n"
            "- When appropriate, generate code that processes or displays the information found in the search results.\n\n"
            "For non-programming questions, respond as a helpful assistant.\n\n"
            "Limitations:\n"
            "- You cannot directly create, modify, or interact with files on the user's system (e.g., you "
            "cannot generate a PDF file directly, open a webpage, or access local files).\n"
            "- If a user asks for something you cannot do directly (like creating a specific file type), "
            "explain your limitation and, if possible, offer a Python code solution that would allow "
            "the user to achieve their goal. For example, if asked for a PDF, you could provide Python "
            "code using a library like ReportLab or FPDF to generate it.\n\n"
            "Focus on providing actionable Python code when appropriate and clearly state your limitations "
            "when a request is outside your capabilities."
        )
        self.venv_dir = ".code_exec_venv"
        self.venv_python_path = None
        self.venv_pip_path = None
        self.tavily_api_key = None
        self.tavily_client = None
        self.setup()

    def _ensure_venv(self):
        """Ensures the virtual environment exists and paths are set."""
        venv_full_path = os.path.join(os.getcwd(), self.venv_dir)

        if os.name == 'nt':
            scripts_dir = "Scripts"
            python_exe = "python.exe"
            pip_exe = "pip.exe"
        else:
            scripts_dir = "bin"
            python_exe = "python"
            pip_exe = "pip"

        self.venv_python_path = os.path.join(venv_full_path, scripts_dir, python_exe)
        self.venv_pip_path = os.path.join(venv_full_path, scripts_dir, pip_exe)

        if not os.path.exists(self.venv_python_path):
            print(f"{Fore.CYAN}Creating virtual environment in '{venv_full_path}'...{Style.RESET_ALL}")
            try:
                subprocess.run([sys.executable, "-m", "venv", venv_full_path], check=True, capture_output=True)
                print(f"{Fore.GREEN}✅ Virtual environment created successfully.{Style.RESET_ALL}")
            except subprocess.CalledProcessError as e:
                print(f"{Fore.RED}❌ Failed to create virtual environment.{Style.RESET_ALL}")
                print(f"{Fore.RED}Error: {e.stderr.decode() if e.stderr else str(e)}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            print(f"{Fore.GREEN}✅ Virtual environment found at '{venv_full_path}'.{Style.RESET_ALL}")

    def setup(self):
        """Set up the Gemini API connection, virtual environment, and Tavily search."""
        print(f"{Fore.CYAN}⚙️  Setting up AI Code Executer...{Style.RESET_ALL}")
        load_dotenv()

        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.api_key = input(f"{Fore.YELLOW}Please enter your Gemini API key: {Style.RESET_ALL}")

        genai.configure(api_key=self.api_key)

        try:
            generation_config = {"temperature": 0.7, "top_p": 0.95, "top_k": 40}
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            self.model = genai.GenerativeModel(
                model_name='gemini-1.5-flash-latest', # Updated to a common recent model
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            self.chat_session = self.model.start_chat(history=[])
            self.chat_session.send_message(self.system_prompt)
            print(f"{Fore.GREEN}✅ Connected to Gemini AI!{Style.RESET_ALL}")

            self._ensure_venv()

            # Setup Tavily Search
            self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
            if self.tavily_api_key:
                if TavilyClient:
                    try:
                        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
                        print(f"{Fore.GREEN}✅ Tavily Search API configured and client initialized.{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}❌ Error initializing Tavily Client: {e}{Style.RESET_ALL}")
                        self.tavily_client = None
                else:
                    print(f"{Fore.YELLOW}⚠️ Tavily Python client ('tavily-python') not installed. Web search functionality will be significantly limited. Please run 'pip install tavily-python'.{Style.RESET_ALL}")
                    self.tavily_client = None
            else:
                print(f"{Fore.YELLOW}⚠️ TAVILY_API_KEY not found in .env file. Web search functionality will be significantly limited. Sign up at tavily.com for an API key and run 'pip install tavily-python'.{Style.RESET_ALL}")
                self.tavily_client = None

        except Exception as e:
            print(f"{Fore.RED}❌ Failed to initialize Gemini API: {e}{Style.RESET_ALL}")
            sys.exit(1)

    def extract_code(self, text):
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else None

    def _extract_dependencies(self, code):
        dependencies = set()
        import_pattern = r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)(?:[.\s]|$)"
        for line in code.splitlines():
            match = re.match(import_pattern, line)
            if match:
                package_name = match.group(1)
                if package_name not in STANDARD_LIBRARY_MODULES:
                    dependencies.add(package_name)
        return list(dependencies)

    def _install_dependency(self, dep, retry_count=0):
        install_name = PACKAGE_INSTALL_MAPPING.get(dep, dep)
        if dep in STANDARD_LIBRARY_MODULES:
            print(f"{Fore.YELLOW}⚠️ Skipping {dep} as it's a standard library/built-in.{Style.RESET_ALL}")
            return True, f"Skipped {dep} (standard/built-in)"
        try:
            print(f"{Fore.YELLOW}Installing {dep} (as {install_name})...{Style.RESET_ALL}")
            install_result = subprocess.run(
                [self.venv_pip_path, "install", install_name],
                capture_output=True, text=True, check=False
            )
            if install_result.returncode == 0:
                print(f"{Fore.GREEN}✅ Successfully installed {dep}.{Style.RESET_ALL}")
                return True, ""
            else:
                error_msg = f"Failed to install {dep}. Pip:\n{install_result.stdout}\n{install_result.stderr}"
                print(f"{Fore.RED}⚠️ {error_msg}{Style.RESET_ALL}")
                if retry_count < self.max_dependency_install_retries:
                    print(f"{Fore.YELLOW}Retrying {dep} (attempt {retry_count+1}/{self.max_dependency_install_retries})...{Style.RESET_ALL}")
                    retry_result = subprocess.run(
                        [self.venv_pip_path, "install", "--no-cache-dir", install_name],
                        capture_output=True, text=True, check=False
                    )
                    if retry_result.returncode == 0:
                        print(f"{Fore.GREEN}✅ Successfully installed {dep} on retry.{Style.RESET_ALL}")
                        return True, ""
                return False, error_msg
        except Exception as e:
            error_msg = f"Error during pip install for {dep}: {e}"
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            return False, error_msg

    def execute_code(self, code, retry_count=0):
        print(f"{Fore.CYAN}⚙️ Executing code{' (retry #'+str(retry_count)+')' if retry_count > 0 else ''}...{Style.RESET_ALL}")
        timeout_seconds = 60
        dependencies = self._extract_dependencies(code)
        failed_dependencies = []
        if dependencies:
            print(f"{Fore.CYAN}Potential dependencies: {', '.join(dependencies)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Installing using pip from '{self.venv_pip_path}'...{Style.RESET_ALL}")
            for dep in dependencies:
                success, error_msg = self._install_dependency(dep)
                if not success: failed_dependencies.append((dep, error_msg))
            if failed_dependencies:
                failure_message = "Failed to install dependencies:\n"
                for dep, error in failed_dependencies: failure_message += f"- {dep}: {error}\n"
                print(f"{Fore.RED}❌ Cannot execute code due to missing dependencies.{Style.RESET_ALL}")
                return failure_message, -1

        print(f"{Fore.BLUE}{'=' * 60}\n{code}\n{'=' * 60}{Style.RESET_ALL}")
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        output_str = ""
        process_result = None
        try:
            print(f"{Fore.YELLOW}Executing with {timeout_seconds}s timeout...{Style.RESET_ALL}")
            process_result = subprocess.run(
                [self.venv_python_path, temp_file_path],
                capture_output=True, text=True, timeout=timeout_seconds
            )
            if process_result.returncode == 0:
                output_str = process_result.stdout
                print(f"{Fore.GREEN}✅ Code executed successfully!{Style.RESET_ALL}")
            else:
                output_str = process_result.stderr
                if not output_str.strip() and process_result.stderr is not None:
                    output_str = f"Error: Code failed (exit code {process_result.returncode}), empty stderr."
                elif process_result.stderr is None:
                    output_str = f"Error: Code failed (exit code {process_result.returncode}), no stderr."
                else:
                    output_str = f"Error: {process_result.stderr}"
                print(f"{Fore.RED}❌ Code execution failed!{Style.RESET_ALL}")
        except subprocess.TimeoutExpired:
            output_str = f"Error: Code execution timed out ({timeout_seconds}s)"
            print(f"{Fore.RED}⏱️ Code execution timed out!{Style.RESET_ALL}")
            if process_result and hasattr(process_result, 'pid'): # process_result might not be defined if timeout occurs very early
                try:
                    if os.name == 'nt':
                        subprocess.run(["taskkill", "/F", "/PID", str(process_result.pid)], capture_output=True, check=False)
                    else:
                        subprocess.run(["kill", "-9", str(process_result.pid)], capture_output=True, check=False)
                except Exception: pass
        except Exception as e:
            output_str = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            print(f"{Fore.RED}❌ Exception during code execution: {e}{Style.RESET_ALL}")
        finally:
            try: os.unlink(temp_file_path)
            except Exception: print(f"{Fore.YELLOW}⚠️ Unable to delete temp file: {temp_file_path}{Style.RESET_ALL}")
        
        return output_str, process_result.returncode if process_result else -1

    def display_welcome(self):
        welcome_text = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🤖 AI Code Executing Agent powered by Gemini           ║
║                                                          ║
║   - Ask any question or request code solutions           ║
║   - Python code will be automatically executed           ║
║   - Type 'upload <filepath> with prompt <your prompt>'   ║
║     to send a file with instructions                     ║
║   - Type 'search <query>' to search the web (requires    ║
║     TAVILY_API_KEY in .env & 'tavily-python' installed)  ║
║   - Type 'exit', 'quit', or 'bye' to end the session     ║
║   - Type 'clear' to clear the conversation history       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
        print(f"{Fore.CYAN}{welcome_text}{Style.RESET_ALL}")

    def _perform_web_search(self, query):
        """Perform a web search using Tavily API and return formatted results and raw list."""
        print(f"{Fore.CYAN}Searching the web for: {query}{Style.RESET_ALL}")
        raw_results_list = []

        if self.tavily_client:
            try:
                print(f"{Fore.CYAN}Using Tavily Search API...{Style.RESET_ALL}")
                # Tavily client search method returns a dictionary.
                # Actual search results are under the 'results' key.
                response_data = self.tavily_client.search(
                    query=query,
                    search_depth="basic",  # "basic" for speed, "advanced" for more detail
                    max_results=self.max_search_results,
                    include_answer=False # Set to True if you want Tavily's summarized answer
                )

                if response_data and 'results' in response_data:
                    tavily_api_results = response_data['results']
                    for res in tavily_api_results:
                        raw_results_list.append({
                            "title": res.get("title", "N/A"),
                            "url": res.get("url", "N/A"),
                            "snippet": res.get("content", "No snippet available.") # Tavily uses 'content'
                        })
                    print(f"{Fore.GREEN}Found {len(raw_results_list)} results using Tavily.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error using Tavily Search API: {e}{Style.RESET_ALL}")
                traceback.print_exc() # For debugging
        else:
            print(f"{Fore.YELLOW}Tavily client not available. Web search skipped.{Style.RESET_ALL}")


        if not raw_results_list:
            # This message will be part of the fallback_summary if Tavily failed or was unavailable
            # _fallback_search_summary now incorporates the original query.
            return self._fallback_search_summary(query), [] 

        # Format the results
        formatted_results_str = f"Search results for '{query}':\n\n"
        for i, result_item in enumerate(raw_results_list, 1):
            formatted_results_str += f"{i}. {result_item['title']}\n"
            formatted_results_str += f"   URL: {result_item['url']}\n"
            formatted_results_str += f"   Snippet: {result_item['snippet']}\n\n"
        
        return formatted_results_str, raw_results_list

    def _fallback_search_summary(self, query):
        """Create a simple summary for when Tavily search fails or is unavailable."""
        return (
            f"I attempted to search for '{query}' using the Tavily API, but couldn't retrieve specific search results. "
            f"This might be due to a missing or invalid TAVILY_API_KEY, the 'tavily-python' library not being installed, "
            f"network issues, or the API returning no results for this query. "
            f"I can still help you with this topic based on my general knowledge. "
            f"What specifically would you like to know or achieve regarding '{query}'?"
        )

    def _handle_search(self, query):
        """Handle search, get AI response based on results."""
        try:
            print(f"{Fore.CYAN}Starting web search for: '{query}'{Style.RESET_ALL}")
            # _perform_web_search returns a formatted string and the raw list of results
            search_results_text, raw_results_list = self._perform_web_search(query)
            
            num_found_results = len(raw_results_list)

            if num_found_results > 0:
                print(f"{Fore.GREEN}✅ Search completed. Found {num_found_results} results to send to AI.{Style.RESET_ALL}")
            else: # Fallback was used or Tavily yielded no results
                 print(f"{Fore.YELLOW}⚠️ Web search did not yield specific results. AI will use general knowledge.{Style.RESET_ALL}")

            # Prepare prompt for AI
            # The search_results_text already contains the fallback message if no results were found.
            search_prompt = (
                f"I've processed a web search for '{query}'.\n"
                f"Search findings:\n{search_results_text}\n\n" # This is the formatted string or fallback message
                f"Based on these findings (or your general knowledge if search failed), "
                f"please provide relevant information. If appropriate, also provide Python code "
                f"that processes or displays information related to '{query}'. "
                f"For example, if the query was about news and a spreadsheet format was desired, "
                f"you might provide code to organize information into a CSV."
            )
            
            print(f"{Fore.YELLOW}Sending search context to AI...{Style.RESET_ALL}")
            response = self.chat_session.send_message(search_prompt)
            response_text = response.text
            print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{response_text}")
            
            code = self.extract_code(response_text)
            if code:
                output, exit_code = self.execute_code(code)
                print(f"{Fore.MAGENTA}Code Output:{Style.RESET_ALL}\n{output}")
                retry_count = 0
                while exit_code != 0 and retry_count < self.max_code_retries and "timed out" not in output.lower():
                    print(f"{Fore.YELLOW}Code failed. Asking AI to fix (retry {retry_count+1}/{self.max_code_retries})...{Style.RESET_ALL}")
                    fix_request = (
                        f"The code generated from the search context for '{query}' failed with:\n{output}\n"
                        f"Please fix it. Ensure your solution handles potential errors."
                    )
                    if "Failed to install" in output:
                        fix_request += "\n\nNote: Dependency installation failed. Suggest alternatives or handle missing dependencies."
                    
                    fix_response = self.chat_session.send_message(fix_request)
                    fix_response_text = fix_response.text
                    print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{fix_response_text}")
                    fixed_code = self.extract_code(fix_response_text)
                    if fixed_code:
                        output, exit_code = self.execute_code(fixed_code, retry_count + 1)
                        print(f"{Fore.MAGENTA}Code Output (Retry {retry_count+1}):{Style.RESET_ALL}\n{output}")
                        retry_count += 1
                    else:
                        print(f"{Fore.RED}❌ AI didn't provide fixed code. Aborting retries.{Style.RESET_ALL}")
                        break
                
                result_message = ""
                if exit_code == 0:
                    result_message = f"Code from search for '{query}' executed. Output:\n{output}\nExplain results or suggest next steps?"
                else:
                    result_message = (f"Code for '{query}' timed out. Suggest more efficient approach?" 
                                      if "timed out" in output.lower() 
                                      else f"After retries, code for '{query}' failed. Error:\n{output}\nSuggest alternative?")
                
                followup = self.chat_session.send_message(result_message)
                print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{followup.text}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error during search handling: {e}{Style.RESET_ALL}")
            traceback.print_exc()

    def _handle_file_upload(self, file_path, user_prompt):
        """Handles file upload and sends content with a prompt to the AI."""
        try:
            # Basic security: resolve path and check it's not going outside current dir too much
            abs_file_path = os.path.abspath(file_path)
            if not abs_file_path.startswith(os.getcwd()):
                # Allow files in subdirectories, but not completely outside project
                # This is a very basic check; for robust security, more is needed.
                # Consider whitelisting directories or using safer path handling.
                 common_prefix = os.path.commonprefix([abs_file_path, os.getcwd()])
                 if common_prefix != os.getcwd(): # If it's not even a subdirectory of cwd
                    print(f"{Fore.RED}❌ File path seems to be outside the allowed working directory.{Style.RESET_ALL}")
                    return

            if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
                print(f"{Fore.RED}❌ File not found or is not a regular file: {file_path}{Style.RESET_ALL}")
                return

            # Read file content (limited size for safety/performance)
            # Max file size to read (e.g., 1MB)
            MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 
            file_size = os.path.getsize(abs_file_path)

            if file_size > MAX_FILE_SIZE_BYTES:
                print(f"{Fore.RED}❌ File is too large ({file_size / (1024*1024):.2f} MB). Maximum allowed size is {MAX_FILE_SIZE_BYTES / (1024*1024):.2f} MB.{Style.RESET_ALL}")
                return
            
            print(f"{Fore.CYAN}Reading file: {file_path}...{Style.RESET_ALL}")
            with open(abs_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            
            if not user_prompt:
                user_prompt = f"The user uploaded this file named '{os.path.basename(file_path)}'. Please analyze it and respond appropriately. If it's code, you can describe what it does. If it's text, summarize it or answer questions about it if I ask."

            # Construct the message to the AI
            # Gemini API has limitations on how files are directly sent.
            # For text-based files, sending content as part of the prompt is typical.
            # For multimodal models, specific file upload mechanisms exist, but here we assume text content.
            
            # Check if the model supports direct file uploads via SDK (more advanced)
            # For now, embedding content in the prompt for text files.
            ai_prompt_with_file = (
                f"{user_prompt}\n\n"
                f"--- Start of uploaded file content ('{os.path.basename(file_path)}') ---\n"
                f"{file_content}\n"
                f"--- End of uploaded file content ('{os.path.basename(file_path)}') ---"
            )
            
            # Truncate if too long for the prompt, though Gemini models have large context windows
            # This is a simple truncation; more sophisticated chunking might be needed for very large files
            # and models with smaller context windows.
            MAX_PROMPT_LENGTH = 30000 # Example, check model's specific limits
            if len(ai_prompt_with_file) > MAX_PROMPT_LENGTH:
                truncate_at = MAX_PROMPT_LENGTH - (len(ai_prompt_with_file) - len(file_content))
                file_content_truncated = file_content[:truncate_at] + "\n[...content truncated...]"
                ai_prompt_with_file = (
                    f"{user_prompt}\n\n"
                    f"--- Start of uploaded file content ('{os.path.basename(file_path)}') (truncated) ---\n"
                    f"{file_content_truncated}\n"
                    f"--- End of uploaded file content ('{os.path.basename(file_path)}') ---"
                )
                print(f"{Fore.YELLOW}⚠️ File content was long and has been truncated to fit the prompt.{Style.RESET_ALL}")

            print(f"{Fore.YELLOW}Sending file content and prompt to AI...{Style.RESET_ALL}")
            response = self.chat_session.send_message(ai_prompt_with_file)
            response_text = response.text
            print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{response_text}")

            # Code execution logic (same as in main loop)
            code = self.extract_code(response_text)
            if code:
                # (Identical code execution and retry logic as in run() method)
                output, exit_code = self.execute_code(code)
                print(f"{Fore.MAGENTA}Code Output:{Style.RESET_ALL}\n{output}")
                
                retry_count = 0
                while exit_code != 0 and retry_count < self.max_code_retries and "timed out" not in output.lower():
                    # ... (same retry logic as in run method) ...
                    print(f"{Fore.YELLOW}Code execution failed. Asking AI to fix the error (retry {retry_count+1}/{self.max_code_retries})...{Style.RESET_ALL}")
                    fix_request = (
                        f"The code you provided (possibly based on the uploaded file '{os.path.basename(file_path)}') failed with the following error:\n\n"
                        f"{output}\n\n"
                        f"Please fix the code and provide a corrected version. Ensure your solution handles the error case."
                    )
                    if "Failed to install the following dependencies:" in output:
                        fix_request += "\n\nNote: There were problems installing required dependencies. Please suggest alternatives or handle the case where these dependencies aren't available."
                    
                    fix_response = self.chat_session.send_message(fix_request)
                    fix_response_text = fix_response.text
                    print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{fix_response_text}")
                    
                    fixed_code = self.extract_code(fix_response_text)
                    if fixed_code:
                        output, exit_code = self.execute_code(fixed_code, retry_count + 1)
                        print(f"{Fore.MAGENTA}Code Output (Retry {retry_count+1}):{Style.RESET_ALL}\n{output}")
                        retry_count += 1
                    else:
                        print(f"{Fore.RED}❌ AI didn't provide a fixed code block. Aborting retry attempts.{Style.RESET_ALL}")
                        break
                
                if exit_code == 0:
                    result_message = (
                        f"The code (possibly based on the uploaded file '{os.path.basename(file_path)}') has been executed successfully. "
                        f"Here is the output:\n{output}\n"
                        f"Is there anything you'd like to explain about these results?"
                    )
                else: # Failed or timed out
                    if "timed out" in output.lower():
                        result_message = (
                            f"The code execution (related to file '{os.path.basename(file_path)}') timed out. "
                            f"This usually happens with long loops or complex tasks. "
                            f"Can you explain the timeout and suggest a more efficient approach?"
                        )
                    else:
                        result_message = (
                            f"After {retry_count} retry attempts, the code (related to file '{os.path.basename(file_path)}') still failed. "
                            f"Final error:\n{output}\n"
                            f"Could you explain the persistent issue?"
                        )
                
                followup = self.chat_session.send_message(result_message)
                print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{followup.text}")

        except FileNotFoundError:
            print(f"{Fore.RED}❌ File not found: {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error handling file upload: {e}{Style.RESET_ALL}")
            traceback.print_exc()


    def run(self):
        self.display_welcome()
        if not self.venv_python_path: self._ensure_venv()

        while True:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
                break
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                self.chat_session = self.model.start_chat(history=[])
                self.chat_session.send_message(self.system_prompt)
                self.display_welcome()
                print(f"{Fore.GREEN}✅ Chat history cleared.{Style.RESET_ALL}")
                continue
            
            if user_input.lower().startswith('search '):
                search_query = user_input[len('search '):].strip()
                if search_query: self._handle_search(search_query)
                else: print(f"{Fore.YELLOW}⚠️ Usage: search <query>{Style.RESET_ALL}")
                continue
            
            if user_input.lower().startswith('upload '):
                upload_parts = user_input[len('upload '):].strip()
                file_path, prompt = (upload_parts.split(' with prompt ', 1) + [None])[:2] if ' with prompt ' in upload_parts.lower() else (upload_parts, None)
                file_path = file_path.strip()
                if prompt: prompt = prompt.strip()
                
                if file_path: self._handle_file_upload(file_path, prompt)
                else: print(f"{Fore.YELLOW}⚠️ Usage: upload <filepath> [with prompt <your prompt>]{Style.RESET_ALL}")
                continue
            
            try:
                print(f"{Fore.YELLOW}AI thinking...{Style.RESET_ALL}")
                response = self.chat_session.send_message(user_input)
                response_text = response.text
                print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{response_text}")
                
                code = self.extract_code(response_text)
                if code:
                    output, exit_code = self.execute_code(code)
                    print(f"{Fore.MAGENTA}Code Output:{Style.RESET_ALL}\n{output}")
                    retry_count = 0
                    while exit_code != 0 and retry_count < self.max_code_retries and "timed out" not in output.lower():
                        print(f"{Fore.YELLOW}Code failed. Asking AI to fix (retry {retry_count+1}/{self.max_code_retries})...{Style.RESET_ALL}")
                        fix_request = (
                            f"The code failed with:\n{output}\n"
                            f"Please fix it. Ensure your solution handles potential errors."
                        )
                        if "Failed to install" in output:
                            fix_request += "\n\nNote: Dependency installation failed. Suggest alternatives or handle missing dependencies."
                        
                        fix_response = self.chat_session.send_message(fix_request)
                        fix_response_text = fix_response.text
                        print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{fix_response_text}")
                        fixed_code = self.extract_code(fix_response_text)
                        if fixed_code:
                            output, exit_code = self.execute_code(fixed_code, retry_count + 1)
                            print(f"{Fore.MAGENTA}Code Output (Retry {retry_count+1}):{Style.RESET_ALL}\n{output}")
                            retry_count += 1
                        else:
                            print(f"{Fore.RED}❌ AI didn't provide fixed code. Aborting retries.{Style.RESET_ALL}")
                            break
                    
                    result_message = ""
                    if exit_code == 0:
                        result_message = f"Code executed. Output:\n{output}\nExplain results or suggest next steps?"
                    else:
                        result_message = (f"Code timed out. Suggest more efficient approach?" 
                                          if "timed out" in output.lower() 
                                          else f"After retries, code failed. Error:\n{output}\nExplain the issue?")
                    
                    followup = self.chat_session.send_message(result_message)
                    print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{followup.text}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error communicating with Gemini API: {e}{Style.RESET_ALL}")
                traceback.print_exc()

if __name__ == "__main__":
    # Before running, ensure you have TAVILY_API_KEY in your .env file
    # and have run: pip install google-generativeai python-dotenv colorama tavily-python
    print(f"{Fore.YELLOW}Make sure you have a .env file with GEMINI_API_KEY and TAVILY_API_KEY.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Required packages: google-generativeai python-dotenv colorama tavily-python{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Install them with: pip install google-generativeai python-dotenv colorama tavily-python{Style.RESET_ALL}")
    
    bot = AiCodeExecuter()
    bot.run()