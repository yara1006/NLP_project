"""LLM API client with SQLite-based prompt-response caching.

Provides DashScope/Qwen API wrapper with automatic caching to reduce
redundant API calls during adversarial sample generation experiments.
"""
from __future__ import annotations

import time
import dashscope
import os
os.environ['NO_PROXY'] = 'aliyuncs.com'
from http import HTTPStatus
import sqlite3
import logging
import threading

class LLMLogSql:
    """SQLite-based prompt-response cache with thread-safe operations.

    Uses a simple key-value table where the prompt is the primary key.
    INSERT OR REPLACE ensures idempotent writes.
    """
    def __init__(self, log_file) -> None:
        self.log_file = log_file
        conn = sqlite3.connect(log_file, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS my_table
                  (Q TEXT PRIMARY KEY, V TEXT)"""
        )
        conn.commit()
        self.lock = threading.Lock()

    def DBQuery(self, Q: str) -> str | None:
        """Query cached response for a given prompt.

        Args:
            Q: The prompt string to look up.

        Returns:
            Cached response string, or None if not found.
        """
        with self.lock:
            conn = sqlite3.connect(self.log_file, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT V FROM my_table WHERE Q=?", (Q,))
            result = cursor.fetchone()
            conn.close()
        return result[0] if result else None

    def DBInsert(self, Q: str, V: str) -> None:
        """Insert or update a prompt-response pair in the cache.

        Args:
            Q: The prompt string (primary key).
            V: The response string to cache.
        """
        with self.lock:
            conn = sqlite3.connect(self.log_file, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO my_table (Q, V) VALUES (?, ?)", (Q, V)
            )
            conn.commit()
            conn.close()

class LLMCall(LLMLogSql):
    """DashScope/Qwen API wrapper with caching and retry logic.

    Extends LLMLogSql to add API call functionality. On each query,
    checks cache first; if miss, calls the API and caches the result.
    """
    log_count: int = 0
    save_count: int = 0

    def __init__(self, log_file: str, API_key: str, version: str, **kwargs) -> None:
        """Initialize the LLM client.

        Args:
            log_file: Path to SQLite cache database.
            API_key: DashScope API key.
            version: Model version string (e.g., "qwen-turbo").
        """
        super().__init__(log_file)
        # API_base is no longer needed for dashscope
        self.version = version
        dashscope.api_key = API_key

    def call(self, prompt: str) -> str:
        """Call the DashScope API with retry logic.

        Retries on API errors or parsing failures with 2-second delays.
        Logs raw responses and exceptions to api_debug.log.

        Args:
            prompt: The input prompt string.

        Returns:
            Response content string, or "ERROR_PARSING_RESPONSE" on failure.
        """
        response = None
        while response is None:
            try:
                # Use dashscope's Generation.call method
                response = dashscope.Generation.call(
                    model=self.version,
                    messages=[{'role': 'user', 'content': prompt}],
                    result_format='message',  # Get message-style output
                    temperature=0.1,
                )

                # Write raw response to a debug file
                with open("api_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"--- RAW RESPONSE ---\n{str(response)}\n--- END RAW RESPONSE ---\n\n")

                if response.status_code != HTTPStatus.OK:
                    # If the API call itself fails
                    logging.warning(f"Dashscope API Error: {response.code} - {response.message}")
                    response = None # Force a retry
                    time.sleep(2)

            except Exception as e:
                logging.warning(f"Exception during API call: {e}")
                # Also log exceptions to the debug file
                with open("api_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"--- API CALL EXCEPTION ---\n{str(e)}\n--- END EXCEPTION ---\n\n")
                response = None # Force a retry
                time.sleep(2)
        
        try:
            # Attempt to parse the response from dashscope format
            content = response.output.choices[0]['message']['content']
            return content
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            # If parsing fails, log the error and the faulty response object
            with open("api_debug.log", "a", encoding="utf-8") as f:
                f.write(f"--- PARSING FAILED ---\n")
                f.write(f"Exception: {str(e)}\n")
                f.write(f"Response object that failed parsing: {str(response)}\n")
                f.write(f"--- END PARSING FAILED ---\n\n")
            # Return a default value that indicates failure
            return "ERROR_PARSING_RESPONSE"

    def query(self, prompt: str) -> str:
        """Query with cache: check cache first, call API on miss.

        Args:
            prompt: The input prompt string.

        Returns:
            Response string from cache or API.
        """
        # Caching logic remains the same
        if save_response := self.DBQuery(prompt):
            self.log_count += 1
            return save_response
        
        response = self.call(prompt)
        # print(f"[DEBUG] Raw LLM response: {response}") # Optional: for deeper debugging
        self.DBInsert(prompt, response)
        self.save_count += 1
        # print(f"Cache hit rate: {self.log_count / (self.save_count + self.log_count)}")
        return response
