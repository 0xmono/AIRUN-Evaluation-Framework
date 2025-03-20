import subprocess
import time
import os
import signal
import atexit
import threading
import logging
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class LlamaServerWrapper:
    def __init__(self, model_path, host="127.0.0.1", port="8080"):
        self.server_process = None
        self.stdout_thread = None
        self.stop_threads = threading.Event()  # Event to signal threads to stop
        self.model_path = model_path
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"

        # Extract a model_id from the filename for reference
        self.model_id = os.path.basename(model_path).split('.')[0]

        # Register cleanup handler for unexpected termination
        atexit.register(self.stop)

    def process_output_stream(self, stream, prefix):
        """Process a stream line by line and log with prefix."""
        while not self.stop_threads.is_set():
            try:
                # Use a timeout to periodically check if we should stop
                line = stream.readline()
                if not line:  # Empty line means EOF reached
                    if self.server_process.poll() is not None:
                        # Process has terminated
                        break
                    # Process still running but no output right now
                    time.sleep(0.1)
                    continue

                logger.info(f"{prefix}{line.strip()}")
            except (ValueError, IOError) as e:
                # Stream closed or other error
                logger.debug(f"Stream processing error: {e}")
                break

    def start(self):
        """Start the llama-server process"""
        if self.server_process:
            print("Server is already running")
            return self

        llama_server_dir = os.environ["LLAMA_SERVER_PATH"]
        llama_server_path = os.path.join(llama_server_dir, "llama-server")

        command = [
            llama_server_path,
            "--model", self.model_path,
            "--host", self.host,
            "--port", self.port,
            "--api-key", "dummy-key"
        ]

        print(f"Starting server with command: {' '.join(command)}")
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1  # Line buffered
        )

        # Reset stop event in case this is being reused
        self.stop_threads.clear()

        # Start threads to handle output asynchronously
        self.stdout_thread = threading.Thread(
            target=self.process_output_stream,
            args=(self.server_process.stdout, "[LLAMA_SERVER] ")
        )

        # Make threads daemon so they exit when main program exits
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        # Wait for server to start
        time.sleep(15)
        print(f"Server started at {self.base_url} with model {self.model_id}")

        # Set environment variables for LangChain
        os.environ["OPENAI_API_KEY"] = "dummy-key"
        os.environ["OPENAI_API_BASE"] = self.base_url

        return self

    def create_llm(self, temperature=0.7, logprobs=None):
        """Create a LangChain LLM instance connected to the running server.

        Args:
            temperature: Controls randomness of output (0.0-2.0)
            logprobs: Number of top logprobs to return per token (None for no logprobs)

        Returns:
            ChatOpenAI: A configured LangChain LLM instance
        """
        if not self.server_process:
            raise RuntimeError("Server is not running. Call start() first.")

        model_kwargs = {}
        if logprobs is not None:
            model_kwargs["logprobs"] = True
            model_kwargs["top_logprobs"] = logprobs

        return ChatOpenAI(
            model_name=self.model_id,
            openai_api_key="dummy-key",
            openai_api_base=self.base_url,
            temperature=temperature,
            model_kwargs=model_kwargs
        )

    def is_running(self):
        """Check if the server is running"""
        return self.server_process is not None and self.server_process.poll() is None

    def stop(self, timeout=2):
        """
        Stop the subprocess and its logging threads.

        Args:
            timeout: Time to wait for threads to join, in seconds
        """
        if self.server_process is None:
            return

        if self.server_process.poll() is None:
            try:
                # Try graceful termination first
                self.server_process.terminate()

                # Give it a moment to terminate
                for _ in range(5):
                    if self.server_process.poll() is not None:
                        break
                    time.sleep(1)

                # Force kill if still running
                if self.server_process.poll() is None:
                    self.server_process.kill()

                print("Server stopped")
            except Exception as e:
                print(f"Error stopping server: {e}")

        # Signal threads to stop
        self.stop_threads.set()

        # Close the pipes to unblock any reads
        try:
            self.server_process.stdout.close()
        except:
            pass

        try:
            self.server_process.stderr.close()
        except:
            pass

        # Wait for threads to finish
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=timeout)
            if self.stdout_thread.is_alive():
                logger.warning("Stdout thread did not terminate properly")

        # Clear threads and process
        self.stdout_thread = None
        self.server_process = None

        logger.info("Subprocess and logging threads stopped")

    def __del__(self):
        """Destructor to ensure server is stopped when the wrapper is garbage collected."""
        self.stop()

    def __enter__(self):
        """Support for 'with' statement - starts the server."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for 'with' statement - stops the server."""
        self.stop()