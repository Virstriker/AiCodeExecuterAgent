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
    "google", "colorama", "dotenv" # Also exclude project-used packages
])

class AiCodeExecuter:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.chat_session = None
        self.max_code_retries = 3  # Maximum number of times to retry executing code after errors
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
        self.setup()
        
    def _ensure_venv(self):
        """Ensures the virtual environment exists and paths are set."""
        venv_full_path = os.path.join(os.getcwd(), self.venv_dir)
        
        if os.name == 'nt':  # Windows
            scripts_dir = "Scripts"
            python_exe = "python.exe"
            pip_exe = "pip.exe"
        else:  # Unix-like
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
        """Set up the Gemini API connection and virtual environment"""
        print(f"{Fore.CYAN}⚙️  Setting up AI Code Executer...{Style.RESET_ALL}")
        
        # Load environment variables from .env file
        load_dotenv()
          # Get API key from environment variable or prompt user
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.api_key = input(f"{Fore.YELLOW}Please enter your Gemini API key: {Style.RESET_ALL}")
            
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40, # Adjusted top_k to a common valid value (e.g., for Flash models)
            } 
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-preview-05-20',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Start chat without system_instruction (add the system prompt as first message)
            self.chat_session = self.model.start_chat(history=[])
            
            # Add system prompt as the first message
            self.chat_session.send_message(self.system_prompt)
            print(f"{Fore.GREEN}✅ Connected to Gemini AI!{Style.RESET_ALL}")
            
            # Ensure virtual environment is set up
            self._ensure_venv()

        except Exception as e:
            print(f"{Fore.RED}❌ Failed to initialize Gemini API: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    def extract_code(self, text):
        """Extract Python code from the AI's response"""
        # Match code blocks with ```python ... ``` format
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0]
        return None
    
    def _extract_dependencies(self, code):
        """Extracts potential third-party dependencies from import statements."""
        dependencies = set()
        # Regex to find 'import package' or 'from package import ...'
        # It captures the first part of the module name.
        import_pattern = r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)(?:[.\s]|$)"
        
        for line in code.splitlines():
            match = re.match(import_pattern, line)
            if match:
                package_name = match.group(1)
                if package_name not in STANDARD_LIBRARY_MODULES:
                    dependencies.add(package_name)
        return list(dependencies)

    def execute_code(self, code, retry_count=0):
        """Execute the provided Python code and return the result"""
        print(f"{Fore.CYAN}⚙️ Executing code{' (retry #'+str(retry_count)+')' if retry_count > 0 else ''}...{Style.RESET_ALL}")
        
        # Define timeout in seconds
        timeout_seconds = 60
        
        dependencies = self._extract_dependencies(code)
        if dependencies:
            print(f"{Fore.CYAN}Found potential dependencies: {', '.join(dependencies)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Attempting to install dependencies using pip from '{self.venv_pip_path}'...{Style.RESET_ALL}")
            for dep in dependencies:
                try:
                    print(f"{Fore.YELLOW}Installing {dep}...{Style.RESET_ALL}")
                    install_result = subprocess.run(
                        [self.venv_pip_path, "install", dep],
                        capture_output=True, text=True, check=False # check=False to handle output manually
                    )
                    if install_result.returncode == 0:
                        print(f"{Fore.GREEN}✅ Successfully installed {dep}.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}⚠️ Failed to install {dep}. Pip output:{Style.RESET_ALL}")
                        print(install_result.stdout)
                        print(install_result.stderr)
                except Exception as e:
                    print(f"{Fore.RED}❌ Error during pip install for {dep}: {e}{Style.RESET_ALL}")

        print(f"{Fore.BLUE}{'=' * 60}\n{code}\n{'=' * 60}{Style.RESET_ALL}")
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        output_str = ""
        try:
            print(f"{Fore.YELLOW}Executing code with a {timeout_seconds} second timeout...{Style.RESET_ALL}")
            # Execute the code in a separate process for safety, using the venv python
            result = subprocess.run(
                [self.venv_python_path, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds  # Ensure timeout is applied
            )
            
            # Get the output
            if result.returncode == 0:
                output_str = result.stdout
                print(f"{Fore.GREEN}✅ Code executed successfully!{Style.RESET_ALL}")
            else:
                output_str = result.stderr
                if not output_str.strip() and result.stderr is not None: # Handle empty stderr
                    output_str = f"Error: Code execution failed with return code {result.returncode} and empty stderr."
                elif result.stderr is None: # Should not happen with capture_output=True but good for robustness
                     output_str = f"Error: Code execution failed with return code {result.returncode} and no stderr captured."
                else:
                    output_str = f"Error: {result.stderr}"

                print(f"{Fore.RED}❌ Code execution failed!{Style.RESET_ALL}")
            
        except subprocess.TimeoutExpired:
            output_str = f"Error: Code execution timed out (exceeded {timeout_seconds} seconds)"
            print(f"{Fore.RED}⏱️ Code execution timed out after {timeout_seconds} seconds!{Style.RESET_ALL}")
            
            # Kill any remaining process (in case the timeout didn't fully terminate it)
            try:
                # Try to find and kill any runaway process
                if os.name == 'nt':  # Windows
                    subprocess.run(["taskkill", "/F", "/PID", str(result.pid)], 
                                  capture_output=True, check=False)
                else:  # Unix/Linux
                    subprocess.run(["kill", "-9", str(result.pid)], 
                                  capture_output=True, check=False)
            except Exception:
                # Ignore errors in the cleanup process
                pass
                
        except Exception as e:
            output_str = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            print(f"{Fore.RED}❌ Exception during code execution: {e}{Style.RESET_ALL}")
        finally:
            # Clean up: delete the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                print(f"{Fore.YELLOW}⚠️ Unable to delete temporary file: {temp_file_path}{Style.RESET_ALL}")
        
        return output_str, result.returncode if 'result' in locals() else -1
    
    def display_welcome(self):
        """Display a welcome message"""
        welcome_text = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🤖 AI Code Executing Agent powered by Gemini           ║
║                                                          ║
║   - Ask any question or request code solutions           ║
║   - Python code will be automatically executed           ║
║   - Type 'upload <filepath> with prompt <your prompt>'   ║
║     to send a file to the AI with specific instructions  ║
║   - Type 'exit', 'quit', or 'bye' to end the session     ║
║   - Type 'clear' to clear the conversation history       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
        print(f"{Fore.CYAN}{welcome_text}{Style.RESET_ALL}")
    
    def _handle_file_upload(self, file_path, prompt=None):
        """Uploads a file to the Gemini API with an optional prompt."""
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.exists(file_path):
                print(f"{Fore.RED}❌ File not found: {file_path}{Style.RESET_ALL}")
                return
            
            if not os.path.isfile(file_path):
                print(f"{Fore.RED}❌ Path is not a file: {file_path}{Style.RESET_ALL}")
                return

            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            
            print(f"{Fore.CYAN}Reading file: {file_path}...{Style.RESET_ALL}")
            
            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(file_extension)
            
            # Read the file based on its type
            if mime_type.startswith('image/') or mime_type.startswith('application/'):
                # Binary file (image, PDF, etc.)
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Create a Gemini-compatible image part
                if mime_type.startswith('image/'):
                    file_part = {"mime_type": mime_type, "data": file_content}
                else:
                    # For other binary files, send as text if possible
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                        # Send as text
                        file_part = file_content
                    except:
                        print(f"{Fore.RED}❌ Cannot process binary file type: {mime_type}{Style.RESET_ALL}")
                        return
            else:
                # Text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                file_part = file_content
            
            # Prepare the message with the file content and prompt
            if prompt:
                user_message = f"I'm uploading a file named '{file_name}'. {prompt}"
            else:
                user_message = f"I'm uploading a file named '{file_name}'. Please analyze it and suggest what I can do with it."
            
            print(f"{Fore.YELLOW}Uploading file to AI with prompt: {prompt or 'No specific prompt'}{Style.RESET_ALL}")
            
            try:
                # For multimodal file uploads like images
                if isinstance(file_part, dict) and 'mime_type' in file_part:
                    response = self.chat_session.send_message([user_message, file_part])
                # For text content
                else:
                    file_message = f"{user_message}\n\nFile Content:\n```\n{file_part}\n```"
                    response = self.chat_session.send_message(file_message)
                    
                response_text = response.text
                
                print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{response_text}")
                
                # Check if the response contains executable code
                code = self.extract_code(response_text)
                if code:
                    output, exit_code = self.execute_code(code)
                    print(f"{Fore.MAGENTA}Code Output:{Style.RESET_ALL}\n{output}")
                    
                    # Handle error cases with retries (except for timeouts)
                    retry_count = 3
                    while exit_code != 0 and retry_count < self.max_code_retries and "timed out" not in output.lower():
                        print(f"{Fore.YELLOW}Code execution failed. Asking AI to fix the error (retry {retry_count+1}/{self.max_code_retries})...{Style.RESET_ALL}")
                        
                        # Send error to AI and ask for a fix
                        fix_request = (
                            f"The code to process the file '{file_name}' failed with the following error:\n\n"
                            f"{output}\n\n"
                            f"Please fix the code and provide a corrected version. Make sure your solution handles the error case."
                        )
                        
                        fix_response = self.chat_session.send_message(fix_request)
                        fix_response_text = fix_response.text
                        print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{fix_response_text}")
                        
                        # Extract the fixed code
                        fixed_code = self.extract_code(fix_response_text)
                        if fixed_code:
                            output, exit_code = self.execute_code(fixed_code, retry_count + 1)
                            print(f"{Fore.MAGENTA}Code Output (Retry {retry_count+1}):{Style.RESET_ALL}\n{output}")
                            retry_count += 1
                        else:
                            print(f"{Fore.RED}❌ AI didn't provide a fixed code block. Aborting retry attempts.{Style.RESET_ALL}")
                            break
                    
                    # Final response after all execution attempts
                    if exit_code == 0:
                        result_message = (
                            f"The code to process the uploaded file '{file_name}' has been executed successfully. "
                            f"Here is the output:\n{output}\n"
                            f"Is there anything you'd like to explain about these results or further actions to take?"
                        )
                    else:
                        if "timed out" in output.lower():
                            result_message = (
                                f"The code to process the uploaded file '{file_name}' timed out. "
                                f"This usually happens when the code has an infinite loop or takes too long to execute. "
                                f"Could you explain what might have caused the timeout and how to avoid it?"
                            )
                        else:
                            result_message = (
                                f"After {retry_count} retry attempts, the code still failed to execute properly. "
                                f"Final error:\n{output}\n"
                                f"Could you explain what might be causing this persistent issue?"
                            )
                    
                    followup = self.chat_session.send_message(result_message)
                    print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{followup.text}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error sending file to AI: {e}{Style.RESET_ALL}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error handling file upload: {e}{Style.RESET_ALL}")
            traceback.print_exc()
    
    def _get_mime_type(self, extension):
        """Determine MIME type based on file extension."""
        mime_types = {
            '.txt': 'text/plain',
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.csv': 'text/csv',
            '.md': 'text/markdown',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        }
        return mime_types.get(extension, 'application/octet-stream')
    
    def run(self):
        """Run the interactive chat loop"""
        self.display_welcome()
        
        # Ensure venv is ready before starting loop if setup failed to complete it (e.g. API key issue first)
        if not self.venv_python_path:
             self._ensure_venv()

        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
                break
                
            # Check for clear command
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                self.chat_session = self.model.start_chat(history=[])
                
                # Re-add system prompt to the new session
                self.chat_session.send_message(self.system_prompt)
                
                self.display_welcome()
                print(f"{Fore.GREEN}✅ Chat history cleared and system prompt re-initialized.{Style.RESET_ALL}")
                continue
            
            # Check for upload command with optional prompt
            if user_input.lower().startswith('upload '):
                # Extract the file path and prompt
                # Format: "upload <filepath> with prompt <your prompt>"
                upload_parts = user_input[len('upload '):].strip()
                
                # Check if there's a "with prompt" part
                if ' with prompt ' in upload_parts.lower():
                    file_path, prompt = upload_parts.split(' with prompt ', 1)
                    file_path = file_path.strip()
                    prompt = prompt.strip()
                else:
                    # No prompt specified
                    file_path = upload_parts
                    prompt = None
                
                if file_path:
                    self._handle_file_upload(file_path, prompt)
                else:
                    print(f"{Fore.YELLOW}⚠️ Please specify a file path. Usage: upload <filepath> [with prompt <your prompt>]{Style.RESET_ALL}")
                continue
            
            # Send message to Gemini AI
            try:
                print(f"{Fore.YELLOW}AI thinking...{Style.RESET_ALL}")
                response = self.chat_session.send_message(user_input)
                response_text = response.text
                
                # Print the response
                print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{response_text}")
                
                # Check if the response contains executable code
                code = self.extract_code(response_text)
                if code:
                    # Execute the code and show the result
                    output, exit_code = self.execute_code(code)
                    print(f"{Fore.MAGENTA}Code Output:{Style.RESET_ALL}\n{output}")
                    
                    # Handle error cases with retries (except for timeouts)
                    retry_count = 0
                    while exit_code != 0 and retry_count < self.max_code_retries and "timed out" not in output.lower():
                        print(f"{Fore.YELLOW}Code execution failed. Asking AI to fix the error (retry {retry_count+1}/{self.max_code_retries})...{Style.RESET_ALL}")
                        
                        # Send error to AI and ask for a fix
                        fix_request = (
                            f"The code failed with the following error:\n\n"
                            f"{output}\n\n"
                            f"Please fix the code and provide a corrected version. Make sure your solution handles the error case."
                        )
                        
                        fix_response = self.chat_session.send_message(fix_request)
                        fix_response_text = fix_response.text
                        print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{fix_response_text}")
                        
                        # Extract the fixed code
                        fixed_code = self.extract_code(fix_response_text)
                        if fixed_code:
                            output, exit_code = self.execute_code(fixed_code, retry_count + 1)
                            print(f"{Fore.MAGENTA}Code Output (Retry {retry_count+1}):{Style.RESET_ALL}\n{output}")
                            retry_count += 1
                        else:
                            print(f"{Fore.RED}❌ AI didn't provide a fixed code block. Aborting retry attempts.{Style.RESET_ALL}")
                            break
                    
                    # Final response after all execution attempts
                    if exit_code == 0:
                        result_message = (
                            f"The code has been executed successfully. Here is the output:\n{output}\n"
                            f"Is there anything you'd like to explain about these results?"
                        )
                    else:
                        if "timed out" in output.lower():
                            result_message = (
                                f"The code execution timed out. "
                                f"This usually happens when the code has an infinite loop or takes too long to execute. "
                                f"Could you explain what might have caused the timeout and how to avoid it?"
                            )
                        else:
                            result_message = (
                                f"After {retry_count} retry attempts, the code still failed to execute properly. "
                                f"Final error:\n{output}\n"
                                f"Could you explain what might be causing this persistent issue?"
                            )
                    
                    followup = self.chat_session.send_message(result_message)
                    print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{followup.text}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error communicating with Gemini API: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    bot = AiCodeExecuter()
    bot.run()
