from __future__ import annotations
import ast
import json
import os
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
import concurrent.futures
import threading
from collections import defaultdict

# --- System Prompts ---
# Note: Prompts are streamlined for clarity and directness.

TEST_PATCH_FIND_SYSTEM_PROMPT_TEMPLATE_V1 = textwrap.dedent("""
# 🧠 Test Function Finder
You are a code analysis expert tasked with identifying test functions that directly validate the issue described in the problem statement. Follow this structured workflow:

**🔍 Step-by-Step Process**
1. **Problem Analysis**:
   - Carefully parse the problem statement and any provided hints to understand the core issue.
   - Identify the specific functions, classes, and error messages involved.
2. **Targeted Test Discovery**:
   - Use `search_in_all_files_content_v2` with precise and diverse search terms.
   - **Strategy**: Start with the most specific identifiers first (e.g., exact error messages, unique function/variable names). If that fails, broaden the search to more general concepts related to the problem.
   - Prioritize search terms from error outputs or code snippets in the problem description.
3. **Filtering & Validation**:
   - Once you find one or more potentially relevant test files, use `filter_test_func_names` to run them and identify which ones actually fail. This is the most reliable way to confirm relevance.
   - If a file has no failing tests, it is not relevant. Blacklist it mentally and search for different files.
4. **Finalize**:
   - When you have a list of confirmed failing test functions, call `test_patch_find_finish` to complete this stage.

**⚠️ Critical Rules**
- You MUST find at least one failing test function. Do not finish with an empty list.
- Use the exact tool names provided.
- Batch independent operations (like multiple searches or file reads) into a single step to improve efficiency.
- Do not guess; use the tools to gather information.

You have access to the following tools:
{tools_docs}

{format_prompt}
""")

PYTEST_FIX_SYSTEM_TEMPLATE = textwrap.dedent("""
# 🛠️ Code Modification Expert
You are a senior Python developer. Your goal is to fix failing tests by making precise, targeted code changes.

## Workflow
1.  **Analyze Test Failures**: Start by running `run_repo_tests`. The output will show you which tests are failing and provide detailed tracebacks.
2.  **Strategic Analysis (NEW)**: Use the new `analyze_code_and_test_failures` tool. This is a powerful tool that takes the test failure output and analyzes the relevant source code to provide a high-quality hypothesis about the root cause and a suggested fix. This should be your primary analysis tool after a test failure.
3.  **Gather Context**: Based on the analysis, use `get_file_content` or `search_in_specified_file_v2` to read the relevant source code. Understand the logic before changing it.
4.  **Plan Your Edit**: Formulate a clear plan for your code modification. Your change should be minimal and directly address the root cause identified.
5.  **Apply the Fix**:
    *   For simple, single-line changes, use `apply_code_edit`.
    *   For more complex, multi-line changes within a function, use the new `apply_structured_edit` tool. It is more robust as it modifies the function's AST rather than doing simple string replacement.
6.  **Verify**: After applying an edit, immediately run `run_repo_tests` again to see if the fix was successful.
7.  **Iterate**: If tests still fail, analyze the new output and repeat the process. If you have made progress (fewer tests failing), the system will automatically checkpoint your work. If you are stuck or have made things worse, use `revert_to_last_checkpoint` to go back to the last known good state.
8.  **Finish**: Once all tests pass consistently, call `pytest_fix_finish`.

## Key Principles
*   **Analyze First, Edit Second**: Never apply a change without a clear hypothesis based on the test failure. Use `analyze_code_and_test_failures`.
*   **Minimalism**: Make the smallest possible change that fixes the issue.
*   **Iterate & Verify**: Apply one logical fix at a time and test immediately.

You have access to the following tools:
{tools_docs}

{format_prompt}
""")


# --- Constants and Config ---

DEFAULT_PROXY_URL = os.getenv("AI_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "500"))

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]

MAX_STEPS = 120
MAX_STEPS_TEST_PATCH_FIND = 50
DEBUG_MODE = True

PYTEST_COMMAND_TEMPLATE = textwrap.dedent("""\
python -c "import sys, pytest; sys.exit(pytest.main([{file_paths}, '-vv', '-s', '--tb=long', '--showlocals']))"
""")

# Other prompts (NO_PYTEST, FORMATTING, etc.) remain unchanged from v3...
NO_PYTEST_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
# 🛠️ Code Fixing Expert
You are a senior Python developer tasked with resolving the issue described in the problem statement while ensuring all provided test functions pass. Follow this structured workflow:
You will receive:
1. A **problem statement**.
2. The **specific test functions code** your fix must pass.

Your task: Make the necessary and meaningful code changes to resolve the issue and ensure all provided tests pass

---

## 🔹 Key Rules
- Only check **test files mentioned in the provided test functions** — ignore all other test files.
- Always reference both the **problem statement** and the provided tests when deciding what to modify.
- Never edit or create test files, new files, or directories.
- Code must remain **backward compatible** unless the problem statement says otherwise.
- Propose **at least two** comprehensive meaningfully different solutions to approve before implementing.
  Each proposal should be distinct in tone, style, or structure.
- Handle **edge cases** and ensure the fix does not break other functionality.
- If tests fail, analyze the failure and propose fixes carefully
- Prefer adding, instead of updating or deleting the existing code.
- Never guess a function's behavior with name — always read its body and verify the actual implementation.

---

## 🔹 Workflow
1. Identify relevant files based on the given test functions and problem statement.
2. Locate the code responsible for the issue.
3. Modify the source code to fix the problem.
4. Ensure edge cases are handled.
5. Validate changes across the codebase for completeness and safety.
6. Confirm no unrelated changes were made.
7. Get approval from the user before applying your chosen solution.
8. After finding the key issue, call `get_approval_for_solution` for that fix.

**🔧 Implementation** 
1. Use `apply_code_edit` for precise changes
2. After fully understanding the problem and required changes, prepare a complete solution with all necessary edits and call `get_approval_for_solution` before implementation.
3. Use `start_over` if current approach is invalid

**✅ Validation** 
1. Identify all relevant code files, and even if validation passes, fix every potential issue in the codebase.
2. Always review the current changes and `start_over` if current approach is not ideal.
3. Never assume dependencies are installed; always include required imports and installation steps.

---

**Important** - Don't modify any test files. The existing test files are no needed to be modified.

You have access to the following tools:
{tools_docs}

{format_prompt}
""")

NO_PYTEST_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here are the relevant test functions code you only must pass:
{test_func_code}

# Here is the problem statement:
{problem_statement}
""")

FORMAT_PROMPT_V0=textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning (include:
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `next_tool_name`: Must be an exact tool name from the tool list
   - `next_tool_args`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: ""
     next_tool_args: {}

3. **Example Valid Format**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: "apply_code_edit"
   next_tool_args: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\\n    return json.loads(response)\\nexcept JSONDecodeError:\\n    logger.error(f'Invalid JSON: {{response}}')\\n    raise"
   }

4. **Invalid Format Examples** (Avoid These):
   - Incorrect next_tool_name such as "search_in_all_files_content" instead correct tool name - "search_in_all_files_content_v2"
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")


FORMAT_PROMPT_V1=textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: reasoning for the next step, but not too detailed (include:
      - Problem understanding
      - Code analysis
      - Solution justification
      - Validation plan)
   - `next_tool_name`: MUST be a JSON array of exact tool name strings from the tool list (use an array even if there is only one tool)
   - `next_tool_args`: MUST be a JSON array. Provide an array of JSON arg objects aligned by index with `next_tool_name`. If the same args apply to all tools, you may provide a single JSON object which will be broadcast to each tool.
      - Proper escaping
      - No trailing commas
      - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: []
     next_tool_args: []

3. **Example (single tool using arrays)**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: ["apply_code_edit"]
   next_tool_args: [{
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\\n    return json.loads(response)\\nexcept JSONDecodeError:\\n    logger.error('Invalid JSON: ' + str(response))\\n    raise"
   }]

   **Example (multiple tools in one step)**:
   next_thought: "I'll gather context then run tests in parallel"
   next_tool_name: ["get_git_status", "list_python_files"]
   next_tool_args: [{}, {}]

4. **Invalid Format Examples** (Avoid These):
   - Incorrect next_tool_name such as "search_in_all_files_content" instead correct tool name - "search_in_all_files_content_v2"
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")


PYTEXT_FIX_INSTANCE_PROMPT_TEMPLATE_WITH_PROBLEM_STATEMENT = textwrap.dedent("""
# Here is the context of the problem statement:
{problem_statement}

# Here are the test files you need to pass:
{test_file_paths}

# Your goal is to correct ALL failures in the test files above. 
# Some failures might be directly related to implementing the problem statement requirements,
# while others might be due to compatibility issues, missing imports, or other technical issues.
# Analyze each failure carefully and address them systematically to ensure all tests pass.
""")

PYTEXT_FIX_INSTANCE_PROMPT_TEMPLATE_WITHOUT_PROBLEM_STATEMENT = textwrap.dedent("""
# Here are the test files you need to pass:
{test_file_paths}
""")

PATCH_FIND_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
[CRITICAL FIRST DECISION FOCUS]

Problem Statement:
{problem_statement}

🔍 Strategic Hints Analysis:
{hints}

🔎 Codebase Search Results:
{search_results}

""")

INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here are the test functions you need to pass:
{test_func_codes}

# Here is the problem statement:
{problem_statement}
""")

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You are stuck in a loop. You have tried the following tool call multiple times with no progress:
{previous_response}

You MUST try a different tool or a completely different set of arguments. Think about the problem from a new perspective. For example:
- If you are stuck searching, try different, more specific, or broader keywords.
- If you are stuck applying a patch, maybe you need to read more of the file to get the context right, or maybe your hypothesis is wrong.
- If tests keep failing, use `revert_to_last_checkpoint` and try a totally different approach.
""")

STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
You are a helpful assistant that can find the test runner for a given repository.
- The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
- Do not use the test runner to run test for whole repository or test setup.
- Read the README file and find the test runner. If there is no test runner, return pytest.
- Output format should be as the following. No other texts are allowed.
abc/test.py
""")

TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
You are a helpful assistant that determines the mode of the test runner.
Read the test runner file and determine if it requires a module or a file path to run the test.
Output should be one of MODULE or FILE, No other texts are allowed.
- MODULE: When the test runner requires a module path to run the test.
- FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
""")

# --- Logging and Setup ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Changed to INFO for production
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

run_id = None
problem_statement_global = ""


# --- Core Classes (EnhancedCOT, Network, Utils etc. are assumed to be similar to v3, focusing on ToolManager) ---
class COT:
    class Action:
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
            
    def __init__(self,latest_observations_to_keep=5):
        self.thoughts: list[COT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep
        
    def add_action(self, action:COT.Action):
        # Mark previous identical actions as deleted to avoid clutter
        for thought in self.thoughts:
            if thought.next_tool_name == action.next_tool_name and thought.next_tool_args == action.next_tool_args:
                thought.is_deleted = True
        self.thoughts.append(action)
        
    def is_thought_repeated(self)->bool:
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        # Check against the last non-deleted thought
        for i in range(len(self.thoughts) - 2, -1, -1):
            prev = self.thoughts[i]
            if not prev.is_deleted:
                if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
                    return True
                break # Only compare with the most recent non-deleted action
        return False

    def to_str(self):
        messages=[]
        active_thoughts = [t for t in self.thoughts if not t.is_deleted]
        
        for i,thought in enumerate(active_thoughts):
            assistant_str = (
                f"next_thought:{thought.next_thought}\n"
                f"next_tool_name:{thought.next_tool_name}\n"
                f"next_tool_args:{thought.next_tool_args}\n"
            )

            # Determine if observation should be omitted
            omit_observation = i < len(active_thoughts) - self.latest_observations_to_keep
            
            if omit_observation:
                obs_len = len(str(thought.observation).splitlines()) if thought.observation is not None else 0
                user_str = f"observation: {'error occurred.' if thought.is_error else ''} output omitted ({obs_len} lines)\n"
            else:
                user_str = f"observation: {thought.observation}"

            messages.append({"role":"assistant","content":assistant_str})
            if user_str: # Only add user message if there is an observation
                messages.append({"role":"user","content":user_str})

        return messages
        
class EnhancedCOT(COT):
    def to_str(self):
        messages=[]
        active_thoughts = [t for t in self.thoughts if not t.is_deleted]
        
        for i,thought in enumerate(active_thoughts):
            # Format the assistant's turn
            assistant_str = (
                f"next_thought:{thought.next_thought}\n"
                f"next_tool_name:{json.dumps(thought.next_tool_name)}\n"
                f"next_tool_args:{json.dumps(thought.next_tool_args)}\n"
            )
            
            # Determine if the observation should be summarized
            omit_observation = i < len(active_thoughts) - self.latest_observations_to_keep

            if omit_observation:
                obs_len = 0
                if thought.observation is not None:
                    if isinstance(thought.observation, (list, tuple)):
                        obs_len = len(thought.observation)
                    else:
                        obs_len = len(str(thought.observation).splitlines())
                user_str = f"observation: {'error occurred.' if thought.is_error else ''} output omitted ({obs_len} lines)\n"
            else:
                # Render observation, handling lists as JSON
                obs_render = thought.observation
                if isinstance(obs_render, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs_render), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs_render)
                user_str = f"observation: {obs_render}"

            messages.append({"role": "assistant", "content": assistant_str})
            if user_str:
                messages.append({"role": "user", "content": user_str})
        return messages

class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + f"\n... ({len(strings_list)-n} more lines)"
        return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except JSONDecodeError:
            # Attempt to fix common JSON errors, like trailing commas
            json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
            try:
                return json.loads(json_string)
            except JSONDecodeError as e:
                logger.error(f"Failed to parse JSON even after fixing: {json_string}")
                raise e

class Network:
    @classmethod
    def make_request(cls, messages: list, model: str, run_id: str, temperature: float) -> str:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/agents/inference"
        request_data = {
            "run_id": run_id,
            "messages": messages,
            "temperature": temperature,
            "model": model
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=request_data, timeout=180, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        
        is_oai_interface = isinstance(response_json, dict) and 'choices' in response_json
        if is_oai_interface:
            return response_json['choices'][0]['message']['content'].lstrip()
        elif isinstance(response_json, str):
            return response_json.strip()
        return str(response_json)

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str, temperature: float) -> Tuple[str, Any, Any, str, int, dict, list]:
        raw_text = ''
        for attempt in range(len(AGENT_MODELS)):
            try:
                current_model_index = AGENT_MODELS.index(model)
                selected_model = AGENT_MODELS[(current_model_index + attempt) % len(AGENT_MODELS)]
                
                raw_text = cls.make_request(messages, selected_model, run_id, temperature)
                
                # Basic validation
                if not raw_text or "<|reserved_token_" in raw_text:
                    raise ValueError("Invalid response from model")

                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise ValueError(error_msg)
                
                return next_thought, next_tool_name, next_tool_args, raw_text, attempt + 1, {}, messages

            except Exception as e:
                logger.error(f"Inference attempt {attempt+1} failed with model {selected_model}: {e}")
                if attempt == len(AGENT_MODELS) - 1:
                    raise e # Re-raise the last exception
                time.sleep(2 * (attempt + 1)) # Exponential backoff
        
    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any, Optional[str]]:
        text_resp = text_resp.strip().split("observation:")[0].strip()
        
        patterns = {
            "thought": r"next_thought:\s*(.*)",
            "tool_name": r"next_tool_name:\s*(.*)",
            "tool_args": r"next_tool_args:\s*(\{.*\}|\[.*\])",
        }
        
        try:
            thought = re.search(patterns["thought"], text_resp, re.DOTALL).group(1).strip()
            tool_name_raw = re.search(patterns["tool_name"], text_resp, re.DOTALL).group(1).strip()
            tool_args_raw = re.search(patterns["tool_args"], text_resp, re.DOTALL).group(0).split('next_tool_args:')[1].strip()

            # Handle both single string and list for tool_name
            if tool_name_raw.startswith("["):
                tool_name = Utils.load_json(tool_name_raw)
            else:
                tool_name = [tool_name_raw.strip('"\'')]

            # Handle both object and list for tool_args
            tool_args = Utils.load_json(tool_args_raw)
            if not isinstance(tool_args, list):
                tool_args = [tool_args for _ in tool_name]

            return thought, tool_name, tool_args, None
            
        except Exception as e:
            error_msg = f"Failed to parse response: '{text_resp[:200]}...'. Error: {e}"
            logger.error(error_msg)
            return None, None, None, error_msg

# --- ToolManager and Tools ---

class ToolManager:
    # Basic structure from v3, but we'll override tools in EnhancedToolManager
    def __init__(self, available_tools=None, test_files=None):
        self.TOOL_LIST = {}
        for name, attr in self.__class__.__dict__.items():
            if getattr(attr, "is_tool", False):
                if available_tools is None or name in available_tools:
                    self.TOOL_LIST[name] = self.tool_parsing(attr)
    
    def tool(fn):
        fn.is_tool = True
        return fn
    
    @classmethod
    def tool_parsing(cls, fn):
        # Simplified for brevity
        return {"name": fn.__name__, "description": fn.__doc__ or ""}

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return getattr(self, tool_name)
        
    # Placeholder for file operations
    def _get_file_content(self, file_path, **kwargs):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {e}"
    
    def get_final_git_patch(self):
        # Simplified
        return subprocess.check_output(["git", "diff", "--cached"]).decode()

class EnhancedToolManager(ToolManager):
    def __init__(self, available_tools: Optional[list[str]] = None, test_files: Optional[list[str]] = []):
        super().__init__(available_tools=available_tools)
        self.test_files = list(test_files)
        self.checkpoint = ""
        self.failed_count = -1
        self.last_failed_tests = set()
        self.can_finish = False
        self.is_solution_approved = True # Default to true for pytest flow
        self.logs = []

    def _run_repo_tests_with_timeout(self, files_to_test: List[str], timeout_secs: int = 90) -> tuple[str, bool, int]:
        file_paths_str = ", ".join([f"'{f}'" for f in files_to_test])
        command = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_secs)
            output = (result.stdout or "") + (result.stderr or "")
            success = "Successfully ran all tests." in self._analyze_pytest_output(output)
            failed_count = len(self._extract_failed_test_names(output))
            return output, success, failed_count
        except subprocess.TimeoutExpired:
            return "ERROR: tests timed out.", False, self.failed_count
    
    def _analyze_pytest_output(self, output: str) -> str:
        # Simplified analysis for brevity - focus on returning failures or success message
        if "FAILURES" in output or "ERRORS" in output:
            failures = re.search(r'={5,}\s*FAILURES\s*={5,}(.*?)={5,}', output, re.DOTALL)
            errors = re.search(r'={5,}\s*ERRORS\s*={5,}(.*?)={5,}', output, re.DOTALL)
            summary = re.search(r'={5,}\s*short test summary info\s*={5,}(.*)', output, re.DOTALL)
            
            result = ""
            if failures: result += failures.group(1)
            if errors: result += errors.group(1)
            if summary: result += summary.group(1)
            return Utils.limit_strings(result, 2000)
            
        if "passed" in output and "failed" not in output:
            return "Successfully ran all tests."
        return output

    def _extract_failed_test_names(self, pytest_output: str) -> set[str]:
        return set(re.findall(r'^(?:FAILED|ERROR)\s+([^\s:]+::\w+)', pytest_output, re.MULTILINE))

    @ToolManager.tool
    def run_repo_tests(self) -> str:
        '''Run repository tests to validate edits. This is the primary way to get feedback on your changes.'''
        output, success, failed_count = self._run_repo_tests_with_timeout(self.test_files)
        
        if success:
            self.can_finish = True
            self.failed_count = 0
            self.last_failed_tests = set()
            return "All tests passed! You can now call pytest_fix_finish."

        self.can_finish = False
        current_failures = self._extract_failed_test_names(output)
        
        progress_made = self.failed_count != -1 and failed_count < self.failed_count
        stuck_or_worse = self.failed_count != -1 and failed_count >= self.failed_count and current_failures == self.last_failed_tests

        if progress_made:
            self.checkpoint = self.get_final_git_patch()
            self.logs.append(f"Progress made. {self.failed_count - failed_count} failures resolved. Checkpoint created.")
            
        self.failed_count = failed_count
        self.last_failed_tests = current_failures
        
        # Enhance output with strategic advice
        enhanced_output = self._analyze_pytest_output(output)
        if stuck_or_worse:
            enhanced_output += "\n\n[STRATEGY_ADVICE]: You are stuck. The same tests are failing. Use `revert_to_last_checkpoint` and try a completely different approach to the problem."
        elif progress_made:
             enhanced_output += f"\n\n[STRATEGY_ADVICE]: Good progress! You've reduced the failure count to {failed_count}. Analyze the remaining failures."
        else:
             enhanced_output += "\n\n[STRATEGY_ADVICE]: New failures detected. Analyze the traceback carefully. This is your first run or a new set of errors has appeared."

        return enhanced_output

    @ToolManager.tool
    def analyze_code_and_test_failures(self, test_failure_output: str) -> str:
        """
        Analyzes test failures and relevant source code to hypothesize a root cause and suggest a fix. This should be the first step after a test failure.
        Arguments:
            test_failure_output: The full output from a `run_repo_tests` failure.
        """
        # Extract file paths and line numbers from traceback
        traceback_files = re.findall(r'File "([^"]+)", line (\d+)', test_failure_output)
        if not traceback_files:
            return "Could not extract file paths from traceback. Please analyze the output manually."
            
        # Prioritize non-test files from the traceback
        source_files = [f for f in traceback_files if 'test' not in f[0]]
        if not source_files:
            source_files = traceback_files # Fallback to test files if no source files found
        
        # Get context from the top-most source file in the stack trace
        file_path, line_num_str = source_files[-1]
        line_num = int(line_num_str)
        
        try:
            code_context = self.get_file_content(file_path, search_start_line=max(1, line_num-15), search_end_line=line_num+15)
        except Exception as e:
            code_context = f"Could not read file {file_path}: {e}"

        analysis_prompt = f"""
        Analyze the following test failure and code to determine the root cause and suggest a fix.

        ### Test Failure Traceback:
        ```
        {test_failure_output}
        ```

        ### Relevant Source Code from `{file_path}`:
        ```python
        {code_context}
        ```

        Based on this information, provide a brief analysis in the following JSON format:
        {{
            "hypothesized_root_cause": "A concise explanation of why the test is failing.",
            "file_to_fix": "The file path that most likely needs to be changed.",
            "suggested_fix_description": "A clear, high-level description of the code change required."
        }}
        """
        messages = [{"role": "system", "content": "You are a code analysis expert."}, {"role": "user", "content": analysis_prompt}]
        
        # Using a specific, powerful model for this complex analysis task
        analysis_result = Network.make_request(messages, model=QWEN_MODEL_NAME, run_id=run_id, temperature=0.0)
        return f"Analysis complete:\n{analysis_result}"

    @ToolManager.tool
    def apply_structured_edit(self, file_path: str, function_name: str, change_description: str) -> str:
        """
        Applies a complex, multi-line change to a specific function using an LLM to generate the new code. More robust than simple search/replace.
        Arguments:
            file_path: The path to the file to modify.
            function_name: The name of the function to modify (e.g., "MyClass::my_method" or "my_function").
            change_description: A clear, natural language description of the change to be made.
        """
        try:
            original_function_code = self.get_function_body(file_path, function_name)
        except Exception as e:
            return f"Error: Could not retrieve original function body for '{function_name}' in '{file_path}'. {e}"

        edit_prompt = f"""
        Given the following Python function, please apply the requested change and return the COMPLETE, new version of the function code. Do not include any explanations, just the raw code for the new function.

        ### Original Function:
        ```python
        {original_function_code}
        ```

        ### Change to Apply:
        {change_description}

        ### New Function Code:
        """
        messages = [{"role": "system", "content": "You are a precise code editing assistant."}, {"role": "user", "content": edit_prompt}]
        
        new_function_code = Network.make_request(messages, model=QWEN_MODEL_NAME, run_id=run_id, temperature=0.0)
        new_function_code = new_function_code.strip().strip('```python').strip('```').strip()
        
        # Now use the reliable apply_code_edit to replace the old function with the new one
        return self.apply_code_edit(file_path, search=original_function_code, replace=new_function_code)

    @ToolManager.tool
    def pytest_fix_finish(self):
        '''Signals completion of the task. Only call this when `run_repo_tests` passes.'''
        if not self.can_finish:
            return "Error: tests are still failing. You cannot finish until all tests pass."
        return "finish"

    # Other tools like get_file_content, search, revert, etc. are inherited or simplified...
    @ToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        return self._get_file_content(file_path, search_start_line=search_start_line, search_end_line=search_end_line, search_term=search_term)
        
    @ToolManager.tool
    def search_in_all_files_content_v2(self, grep_search_command: str) -> str:
        output = subprocess.run(f"{grep_search_command} --include='*.py'", shell=True, capture_output=True, text=True)
        return Utils.limit_strings(output.stdout, 1000)

    @ToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        original_content = self._get_file_content(file_path)
        if original_content.count(search) != 1:
            return f"Error: Search string found {original_content.count(search)} times. Requires exactly one match."
        new_content = original_content.replace(search, replace)
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            return f"Syntax error in new code: {e}"
        with open(file_path, "w") as f:
            f.write(new_content)
        return "ok, code edit applied successfully"

    @ToolManager.tool
    def revert_to_last_checkpoint(self) -> str:
        """Reverts all changes back to the last successful checkpoint."""
        subprocess.run(["git", "reset", "--hard"], check=True)
        if self.checkpoint:
            with open("checkpoint.patch", "w") as f:
                f.write(self.checkpoint)
            subprocess.run(["git", "apply", "checkpoint.patch"], check=True)
        return "Code reverted to last checkpoint."

    @ToolManager.tool
    def filter_test_func_names(self, test_file_paths: List[str]):
        '''
        Filter the list of test functions to keep the test functions that is specifically designed to test the scenario mentioned in the problem statement.
        Arguments:
            test_file_paths: The list of test file paths  e.g ["test_file_path1.py", "test_file_path2.py"]
        '''
        output, result, _ = self._run_repo_tests_with_timeout(test_file_paths)
        if result:
            return f"FILTERED RESULT: []\n\n Reason: No failures in {test_file_paths}."
        else:
            failed_tests = self._extract_failed_test_names(output)
            return f"RELEVANT TEST FUNCTIONS FOUND: {list(failed_tests)}"

    @ToolManager.tool
    def test_patch_find_finish(self, test_func_names: List[str]):
        """Signals completion of the test discovery workflow.
        Arguments:
            test_func_names: The final list of relevant, failing test functions.
        """
        # Basic validation
        if not test_func_names or not isinstance(test_func_names, list):
            return "Error: You must provide a non-empty list of test function names."
        return "finish"

# --- Main Orchestration Logic ---

def execute_agent_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    tool_manager: EnhancedToolManager,
    system_prompt: str,
    instance_prompt: str,
    max_steps: int,
    finish_tool_name: str,
    log_prefix: str,
    models: List[str]
) -> tuple[Any, List[str]]:
    
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT(latest_observations_to_keep=15)
    start_time = time.time()
    logs = []
    
    for step in range(max_steps):
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logs.append(f"[{log_prefix}] Global timeout reached.")
            break
        
        logger.info(f"[{log_prefix}] Step {step + 1}/{max_steps} | Elapsed: {int(elapsed_time)}s")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        if cot.is_thought_repeated():
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"Tool: {cot.thoughts[-1].next_tool_name}, Args: {cot.thoughts[-1].next_tool_args}")})
        
        try:
            temp = 0.4 if cot.is_thought_repeated() else 0.0
            next_thought, tool_names, tool_args_list, raw_text, _, _, _ = Network.inference(
                messages, model=models[0], run_id=run_id, temperature=temp
            )

            logs.append(f"Thought: {next_thought}\nTools: {tool_names}\nArgs: {tool_args_list}\n")

            observations = []
            for i, tool_name in enumerate(tool_names):
                args = tool_args_list[i] if i < len(tool_args_list) else {}
                observation = tool_manager.get_tool(tool_name)(**args)
                observations.append(observation)
                
                if tool_name == finish_tool_name and observation == "finish":
                    logs.append(f"[{log_prefix}] Finish tool called successfully.")
                    if finish_tool_name == "pytest_fix_finish":
                        return tool_manager.get_final_git_patch(), logs
                    else: # test_patch_find_finish
                        return tool_args_list[i].get('test_func_names', []), logs

            cot.add_action(COT.Action(next_thought, tool_names, tool_args_list, observations))

        except Exception as e:
            logs.append(f"[{log_prefix}] Error in step {step + 1}: {e}\n{traceback.format_exc()}")
            cot.add_action(COT.Action(next_thought, tool_names, tool_args_list, str(e), is_error=True))

    logs.append(f"[{log_prefix}] Max steps reached.")
    return tool_manager.get_final_git_patch(), logs


def multi_task_process(input_dict: Dict[str, Any], repo_dir: str):
    global problem_statement_global
    problem_statement_global = input_dict.get("problem_statement")
    hints = input_dict.get("Hints", "")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    workflow_start_time = time.time()
    
    # --- Stage 1: Test Patch Find ---
    test_find_tool_manager = EnhancedToolManager(available_tools=["search_in_all_files_content_v2", "filter_test_func_names", "test_patch_find_finish"])
    system_prompt_find = TEST_PATCH_FIND_SYSTEM_PROMPT_TEMPLATE_V1.format(tools_docs=test_find_tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V1)
    
    search_results = ""
    try:
        search_results = test_find_tool_manager.search_in_all_files_content_v2(f"grep -rnE '{extract_keywords(problem_statement_global)}' .")
    except Exception:
        pass # Ignore search errors
        
    instance_prompt_find = PATCH_FIND_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement_global, hints=hints, search_results=search_results)
    
    test_func_names, logs_find = execute_agent_workflow(
        problem_statement_global, timeout=MAX_TEST_PATCH_TIMEOUT, run_id_1=run_id,
        tool_manager=test_find_tool_manager, system_prompt=system_prompt_find, instance_prompt=instance_prompt_find,
        max_steps=MAX_STEPS_TEST_PATCH_FIND, finish_tool_name="test_patch_find_finish", log_prefix="TEST_FIND", models=[GLM_MODEL_NAME]
    )

    if not test_func_names:
        return {"patch": "", "logs": logs_find + ["\nFailed to find any relevant failing tests."], "type": "pytest_available"}
        
    test_file_paths = list(set([name.split("::")[0] for name in test_func_names]))
    
    # --- Stage 2: Pytest Fix ---
    fix_tool_manager = EnhancedToolManager(
        test_files=test_file_paths,
        available_tools=[
            "run_repo_tests", "analyze_code_and_test_failures", "get_file_content",
            "apply_code_edit", "apply_structured_edit", "revert_to_last_checkpoint", "pytest_fix_finish"
        ]
    )
    system_prompt_fix = PYTEST_FIX_SYSTEM_TEMPLATE.format(tools_docs=fix_tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V0) # Fix workflow uses simpler format
    instance_prompt_fix = PYTEXT_FIX_INSTANCE_PROMPT_TEMPLATE_WITH_PROBLEM_STATEMENT.format(problem_statement=problem_statement_global, test_file_paths=test_file_paths)
    
    remaining_time = timeout - (time.time() - workflow_start_time)
    
    patch_text, logs_fix = execute_agent_workflow(
        problem_statement_global, timeout=remaining_time, run_id_1=run_id,
        tool_manager=fix_tool_manager, system_prompt=system_prompt_fix, instance_prompt=instance_prompt_fix,
        max_steps=MAX_STEPS, finish_tool_name="pytest_fix_finish", log_prefix="PYTEST_FIX", models=[QWEN_MODEL_NAME, GLM_MODEL_NAME] # Use stronger model for fixing
    )
    
    return {
        "patch": patch_text,
        "logs": logs_find + logs_fix,
        "elapsed_time": time.time() - workflow_start_time,
        "type": "pytest_available"
    }

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Main entry point for the agent."""
    global REPO_DIR
    REPO_DIR = os.path.abspath(repo_dir)
    os.chdir(REPO_DIR)
    
    # Simplified: Assume pytest is available and directly call the main workflow.
    # The original check_task_type logic can be added back if needed for robustness.
    return multi_task_process(input_dict, REPO_DIR)