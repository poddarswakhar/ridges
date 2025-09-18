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

# --- System Prompts (Updated for New Workflow) ---

TEST_PATCH_FIND_SYSTEM_PROMPT_TEMPLATE_V0 = textwrap.dedent("""
# 🧠 Test Function Discovery Expert
You are a systematic test discovery specialist. Your mission is to identify test functions that validate the specific issue described in the problem statement.
## 📋 STRUCTURED WORKFLOW (Follow this exact sequence):
### Step 1: 🔍 Problem Analysis
- Parse the problem statement carefully and read "Hints" if present.
- Identify affected functions, classes, and modules.
### Step 2: 🗂 Test File Discovery
**Use `search_in_all_files_content_v2` to find relevant test files.**
- Use distinctive variables, literals, and parts of error messages as keywords.
### Step 3: 📝 Extract Relevant Existing Test Functions
**For each discovered test file, use `find_relevant_tests_in_file` to identify relevant functions.**
- This tool will help you determine which functions are most relevant to the problem.
### Step 4: ✅ Complete Discovery
**Call `test_patch_find_finish_v0` ONLY when you have found relevant tests.**
- You MUST find at least one relevant test function before finishing.
## 🎯 CRITICAL SUCCESS CRITERIA:
- Find test functions that directly validate the problematic behavior.
- Be thorough in file discovery before function extraction.
- Keep searching until you find at least ONE relevant test.
- **NEVER** call `test_patch_find_finish_v0` with an empty list.
## 🛠 Available Tools:
{tools_docs}
{format_prompt}
""")

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
You have access to the following tools:-
{tools_docs}
{format_prompt}
""")

PYTEST_FIX_SYSTEM_TEMPLATE = textwrap.dedent("""
# 🛠️ Code Modification Expert
You are a senior Python developer. Your goal is to fix failing tests by making precise, targeted code changes.

## Workflow
1.  **Analyze Test Failures**: Start by running `run_repo_tests`. The output will show you which tests are failing and provide detailed tracebacks.
2.  **Strategic Analysis (NEW)**: Use the new `analyze_code_and_test_failures` tool. This is a powerful tool that takes the test failure output and analyzes the relevant source code to provide a high-quality hypothesis about the root cause and a suggested fix. This should be your primary analysis tool after a test failure.
3.  **Gather Context**: Based on the analysis, use `get_file_content` to read the relevant source code. Understand the logic before changing it.
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

# --- Constants, Configs, and other boilerplate from v3 file... ---
# (Assuming the rest of the prompts and constants from the v3 file are here)

FORMAT_PROMPT_V0=textwrap.dedent("""
**📝 Response Format Requirements**
1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning.
   - `next_tool_name`: Must be an exact tool name.
   - `next_tool_args`: Valid JSON.
""")

FORMAT_PROMPT_V1=textwrap.dedent("""
**📝 Response Format Requirements**
1. **Strict Triplet Format**:
   - `next_thought`: Reasoning for the next step.
   - `next_tool_name`: MUST be a JSON array of tool names.
   - `next_tool_args`: MUST be a JSON array of objects, aligned with `next_tool_name`.
""")

PYTEXT_FIX_INSTANCE_PROMPT_TEMPLATE_WITH_PROBLEM_STATEMENT = textwrap.dedent("""
# Here is the context of the problem statement:
{problem_statement}
# Here are the test files you need to pass:
{test_file_paths}
# Your goal is to correct ALL failures in these test files.
""")

PYTEXT_FIX_INSTANCE_PROMPT_TEMPLATE_WITHOUT_PROBLEM_STATEMENT = textwrap.dedent("""
# Here are the test files you need to pass:
{test_file_paths}
""")

PATCH_FIND_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
Problem Statement:
{problem_statement}
Hints: {hints}
Codebase Search Results:
{search_results}
""")

INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Test functions to pass:
{test_func_codes}
# Problem statement:
{problem_statement}
""")

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You are stuck in a loop. You have tried the following tool call multiple times with no progress:
{previous_response}
You MUST try a different tool or a completely different set of arguments. For example:
- If searching, try different keywords.
- If applying a patch fails, read more code to get context.
- If tests keep failing, use `revert_to_last_checkpoint` and try a new approach.
""")

STOP_INSTRUCTION=textwrap.dedent("Generate only a SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args`.")

DEFAULT_PROXY_URL = os.getenv("AI_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "500"))

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS=[GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]

MAX_STEPS = 120
MAX_STEPS_TEST_PATCH_FIND = 50
DEBUG_MODE=True

PYTEST_COMMAND_TEMPLATE = textwrap.dedent("""\
python -c "import sys, pytest; sys.exit(pytest.main([{file_paths}, '-vv', '-s', '--tb=long', '--showlocals']))"
""")

# ... (all other prompts and constants from v3 file are assumed here)
logger = logging.getLogger(__name__)
# ... (logging setup)

# --- Core Classes (EnhancedCOT, Network, Utils, etc.) ---
# These classes are largely the same as v3, but included for completeness.
class EnhancedCOT:
    # ... (Implementation from v3)
    pass
# ... (Other helper classes like SmartCache, Parallel executors, etc.)

# --- ToolManager and Tools ---
class EnhancedToolManager(ToolManager):
    # This class will be augmented with new tools and logic.
    # The original methods from v3 are preserved.

    def __init__(self, available_tools: Optional[list[str]] = None, test_files: Optional[list[str]] = []):
        super().__init__(available_tools, test_files)
        # Existing initializations from v3
        self.checkpoint = ""
        self.failed_count = -1
        self.last_failed_tests = set()
        self.can_finish = False

    def _run_repo_tests_with_timeout(self, files_to_test: List[str], timeout_secs: int = 90) -> tuple[str, bool, int]:
        # Helper function to execute tests
        file_paths_str = ", ".join([f"'{f}'" for f in files_to_test])
        command = self.pytest_command_template.format(file_paths=file_paths_str)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_secs)
            output = (result.stdout or "") + (result.stderr or "")
            failed_tests = self._extract_failed_test_names(output)
            success = not bool(failed_tests)
            return output, success, len(failed_tests)
        except subprocess.TimeoutExpired:
            return "ERROR: tests timed out.", False, self.failed_count if self.failed_count != -1 else 1

    def _extract_failed_test_names(self, pytest_output: str) -> set[str]:
        """Extracts FAILED test function names from pytest output."""
        # This uses a regex that is more robust for various pytest formats
        return set(re.findall(r'^(?:FAILURES|ERRORS)\s*_{2,}\s*(.*?)\s*_{2,}', pytest_output, re.MULTILINE | re.DOTALL))

    @ToolManager.tool
    def run_repo_tests(self, timeout_secs: int = 90) -> str:
        '''
        Run repository tests for the selected test files to validate edits.
        This tool now includes automatic checkpointing on progress and strategic advice if stuck.
        '''
        output, success, failed_count = self._run_repo_tests_with_timeout(self.test_files, timeout_secs)

        if success:
            self.can_finish = True
            self.failed_count = 0
            self.last_failed_tests = set()
            return "Successfully ran all tests. You can now call pytest_fix_finish."

        self.can_finish = False
        current_failures = self._extract_failed_test_names(output)
        
        # Check for progress or regression
        progress_made = self.failed_count != -1 and failed_count < self.failed_count
        stuck_or_worse = self.failed_count != -1 and failed_count >= self.failed_count and current_failures == self.last_failed_tests

        if progress_made:
            self.checkpoint = self.get_final_git_patch()
            self.logs.append(f"Progress made. {self.failed_count - failed_count} failures resolved. Checkpoint created.")
            
        self.failed_count = failed_count
        self.last_failed_tests = current_failures
        
        # Enhance output with strategic advice
        enhanced_output = self.analyze_pytest_output(output)
        if stuck_or_worse:
            enhanced_output += "\n\n[STRATEGY_ADVICE]: You are stuck. The same tests are failing. Use `revert_to_last_checkpoint` and try a completely different approach to the problem."
        elif progress_made:
             enhanced_output += f"\n\n[STRATEGY_ADVICE]: Good progress! You've reduced the failure count to {failed_count}. Analyze the remaining failures."
        else:
             enhanced_output += "\n\n[STRATEGY_ADVICE]: New failures detected. Analyze the traceback carefully. This is your first run or a new set of errors has appeared."

        return enhanced_output

    # --- NEW TOOLS START HERE ---
    @ToolManager.tool
    def analyze_code_and_test_failures(self, test_failure_output: str) -> str:
        '''
        Analyzes test failures and relevant source code to hypothesize a root cause and suggest a fix. This should be the first step after a test failure.
        Arguments:
            test_failure_output: The full output from a `run_repo_tests` failure.
        '''
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
            # Get focused context around the error line
            code_context = self.get_file_content(
                file_path=file_path, 
                search_start_line=max(1, line_num - 15),
                search_end_line=line_num + 15
            )
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
        analysis_result = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, run_id=run_id, temperature=0.0)
        return f"Analysis complete:\n{analysis_result}"

    @ToolManager.tool
    def apply_structured_edit(self, file_path: str, function_name: str, change_description: str) -> str:
        '''
        Applies a complex, multi-line change to a specific function using an LLM to generate the new code. More robust than simple search/replace.
        Arguments:
            file_path: The path to the file to modify.
            function_name: The name of the function to modify (e.g., "MyClass::my_method" or "my_function").
            change_description: A clear, natural language description of the change to be made.
        '''
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
        
        new_function_code = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, run_id=run_id, temperature=0.0)
        new_function_code = new_function_code.strip().strip('```python').strip('```').strip()
        
        # Now use the reliable apply_code_edit to replace the old function with the new one
        return self.apply_code_edit(file_path, search=original_function_code, replace=new_function_code)
    # --- END OF NEW TOOLS ---
    
    # ... (All other tools from v3, like get_file_content, apply_code_edit, filter_test_func_names, etc., are preserved here)
    # For brevity, I am not re-pasting all of them, but they are part of this class.

# --- Main Orchestration Logic ---
# The main workflow functions (execute_agent_workflow, multi_task_process, agent_main, etc.)
# are preserved from v3 but will now be able to call the new tools via the updated prompt.
# ... (The rest of the agent_main, multi_task_process, and other functions from v3 would follow here)