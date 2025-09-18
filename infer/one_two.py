0a1,2
> # “Trying Is Half the Battle (The Other Half Is Trying Again)” v2.1
> 
24d25
< #123
447,448c448,449
< DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
< MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "500"))
---
> DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))
> MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))
452d452
< DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
454c454,455
< AGENT_MODELS=[GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
---
> DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
> AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]
456,457c457,458
< MAX_STEPS = 120
< MAX_STEPS_TEST_PATCH_FIND = 50
---
> MAX_STEPS = 250
> MAX_STEPS_TEST_PATCH_FIND = 100
459,466d459
< 
< # 🚀 Enhanced Accuracy Algorithm Configuration
< SELF_CONSISTENCY_CONFIG = {
<     'DEFAULT_NUM_PATHS': 5,
<     'DEFAULT_CONSENSUS_THRESHOLD': 0.6,
<     'MAX_EXECUTION_TIME': 30,  # seconds
<     'ENABLE_ADAPTIVE_PATHS': True
< }
468,474d460
< INTELLIGENT_SEARCH_CONFIG = {
<     'DEFAULT_FUSION_METHOD': 'weighted',
<     'MAX_SEARCH_STRATEGIES': 5,
<     'SEARCH_TIMEOUT': 20,  # seconds per strategy
<     'ENABLE_CONTEXT_ANALYSIS': True,
<     'ENABLE_ADAPTIVE_ROUTING': True
< }
476,483d461
< # Combined accuracy improvement estimation
< EXPECTED_ACCURACY_IMPROVEMENT = {
<     'self_consistency': 0.25,  # +25%
<     'intelligent_search': 0.15,  # +15%
<     'combined': 0.40,  # +40% (synergistic effect)
<     'confidence_threshold': 0.8
< }
< 
532c510,543
< # Enhanced caching and timeout system
---
> class TemperatureManager:
>     """Manages dynamic temperature adjustment based on COT failure patterns"""
>     
>     def __init__(self, initial_temp: float = 0.0, failure_temp: float = 0.1):
>         self.initial_temp = initial_temp
>         self.failure_temp = failure_temp
>         self.current_temp = initial_temp
>         self.consecutive_failures = 0
>         self.failure_threshold = 1  # Adjust after first failure
>         self.reset_after_success = True
>         
>     def on_failure(self) -> float:
>         """Called when a COT inference fails. Returns new temperature."""
>         self.consecutive_failures += 1
>         if self.consecutive_failures >= self.failure_threshold:
>             self.current_temp = self.failure_temp
>         return self.current_temp
>         
>     def on_success(self) -> float:
>         """Called when a COT inference succeeds. Returns new temperature."""
>         if self.reset_after_success:
>             self.consecutive_failures = 0
>             self.current_temp = self.initial_temp
>         return self.current_temp
>         
>     def get_current_temp(self) -> float:
>         """Get current temperature setting."""
>         return self.current_temp
>         
>     def reset(self):
>         """Reset to initial state."""
>         self.current_temp = self.initial_temp
>         self.consecutive_failures = 0
> 
710,735d720
<     def search_multiple_directories_parallel(self, directories: List[str], search_term: str) -> Dict[str, str]:
<         """Search the same term across multiple directories in parallel"""
<         
<         def search_directory(directory: str) -> tuple[str, str]:
<             try:
<                 result = self.tool_manager.search_recurive_in_all_files_in_directory(
<                     directory_path=directory,
<                     search_term=search_term
<                 )
<                 return directory, result
<             except Exception as e:
<                 return directory, f"Error searching in '{directory}': {e}"
<         
<         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(directories), 3)) as executor:
<             future_to_dir = {
<                 executor.submit(search_directory, directory): directory 
<                 for directory in directories
<             }
<             
<             results = {}
<             for future in concurrent.futures.as_completed(future_to_dir):
<                 directory, result = future.result()
<                 results[directory] = result
<         
<         return results
< 
764,793c749
<     def apply_multiple_edits_parallel(self, edits: List[Dict[str, Any]]) -> Dict[str, str]:
<         """Apply multiple code edits in parallel"""
<         
<         def apply_single_edit(edit: Dict[str, Any]) -> tuple[str, str]:
<             try:
<                 file_path = edit['file_path']
<                 search = edit['search']
<                 replace = edit['replace']
<                 
<                 result = self.tool_manager.apply_code_edit(
<                     file_path=file_path,
<                     search=search,
<                     replace=replace
<                 )
<                 return file_path, result
<             except Exception as e:
<                 return edit.get('file_path', 'unknown'), f"Error applying edit: {e}"
<         
<         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(edits), 3)) as executor:
<             future_to_edit = {
<                 executor.submit(apply_single_edit, edit): edit 
<                 for edit in edits
<             }
<             
<             results = {}
<             for future in concurrent.futures.as_completed(future_to_edit):
<                 file_path, result = future.result()
<                 results[file_path] = result
<         
<         return results
---
> 
798,833d753
<     
<     def execute_with_dependencies(self, problem_statement: str, test_func_names: List[str]) -> Dict[str, Any]:
<         """Execute operations in parallel where possible, respecting dependencies"""
<         
<         # Phase 1: Independent operations (can run in parallel)
<         phase1_tasks = {
<             'file_listing': lambda: self.tool_manager.list_python_files(),
<             'git_status': lambda: self.tool_manager.get_git_status(),
<             'git_branches': lambda: self.tool_manager.get_git_branches()
<         }
<         
<         phase1_results = self._execute_parallel(phase1_tasks)
<         
<         # Phase 2: Operations that depend on Phase 1 results
<         python_files = phase1_results.get('file_listing', '').split('\n')
<         relevant_files = [f for f in python_files if f.strip()]
<         
<         phase2_tasks = {}
<         for file_path in relevant_files[:5]:  # Limit to first 5 files
<             phase2_tasks[f'analyze_{file_path}'] = lambda fp=file_path: self._analyze_file(fp)
<         
<         phase2_results = self._execute_parallel(phase2_tasks)
<         
<         # Phase 3: Operations that depend on test functions
<         phase3_tasks = {}
<         for test_func in test_func_names:
<             file_path, func_name = test_func.split(" - ")
<             phase3_tasks[f'test_analysis_{func_name}'] = lambda fp=file_path, fn=func_name: self._analyze_test(fp, fn)
<         
<         phase3_results = self._execute_parallel(phase3_tasks)
<         
<         return {
<             'phase1': phase1_results,
<             'phase2': phase2_results,
<             'phase3': phase3_results
<         }
835,873d754
<     def _execute_parallel(self, tasks: Dict[str, callable]) -> Dict[str, Any]:
<         """Execute a dictionary of tasks in parallel"""
<         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
<             future_to_task = {
<                 executor.submit(task_func): task_name 
<                 for task_name, task_func in tasks.items()
<             }
<             
<             results = {}
<             for future in concurrent.futures.as_completed(future_to_task):
<                 task_name = future_to_task[future]
<                 try:
<                     result = future.result(timeout=60)
<                     results[task_name] = result
<                 except Exception as e:
<                     results[task_name] = f"Error: {e}"
<         
<         return results
<     
<     def _analyze_file(self, file_path: str) -> Dict[str, Any]:
<         """Analyze a single file with multiple tools"""
<         try:
<             return {
<                 'content': self.tool_manager.get_file_content(file_path, limit=1000),
<                 'smells': self.tool_manager.detect_code_smells(file_path),
<                 'quality': self.tool_manager.get_code_quality_metrics(file_path)
<             }
<         except Exception as e:
<             return {'error': str(e)}
<     
<     def _analyze_test(self, file_path: str, func_name: str) -> Dict[str, Any]:
<         """Analyze a test function"""
<         try:
<             return {
<                 'body': self.tool_manager.get_function_body(file_path, func_name),
<                 'coverage': self.tool_manager.analyze_test_coverage([f"{file_path} - {func_name}"])
<             }
<         except Exception as e:
<             return {'error': str(e)}
901c782,783
<         
---
>         # Check if the last thought is the same as the previous thought.
>         # If there are less than 2 thoughts, skip (return False).
1113a996,998
>     # Global temperature manager for all network instances
>     _temperature_manager = TemperatureManager()
>         
1159c1044
<     def make_request(cls,messages:list,attempt:int=10, temperature: float = 0.0)->str:
---
>     def make_request(cls,messages:list,attempt:int=10)->str:
1166c1051
<                 "temperature": temperature,
---
>                 "temperature": cls._temperature_manager.get_current_temp(),
1193,1194c1078
<                             base_delay: float = 2.0,
<                             temperature: float = 0.0) -> str:
---
>                             base_delay: float = 2.0) -> str:
1203c1087
<                 raw_text=cls.make_request(messages,attempt=attempt, temperature=temperature)
---
>                 raw_text=cls.make_request(messages,attempt=attempt)
1318c1202
<     def inference(cls, messages: List[Dict[str, Any]], run_id: str = "1",return_json:bool=False, temperature: float = 0.0) -> dict:
---
>     def inference(cls, messages: List[Dict[str, Any]], run_id: str = "1",return_json:bool=False) -> dict:
1327a1212,1213
>             # Ignore assistant placeholders that only carry the internal
>             # ``tool_call`` and have no visible content.
1336c1222
<         next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, temperature=temperature)
---
>         next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs)
1394a1281,1282
>     # Inherit the temperature manager from parent class
>     
1416c1304
<     def make_request(cls,messages:list,model:str,attempt:int=10, temperature: float = 0.0)->str:
---
>     def make_request(cls,messages:list,model:str,attempt:int=10)->str:
1423c1311
<                 "temperature": temperature,
---
>                 "temperature": cls._temperature_manager.get_current_temp(),
1454,1455c1342
<                             base_delay: float = 1.0,
<                             temperature: float = 0.0) -> str:
---
>                             base_delay: float = 1.0) -> str:
1465c1352
<                 raw_text=cls.make_request(messages,model=AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)], temperature=temperature)
---
>                 raw_text=cls.make_request(messages,model=AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)])
1511c1398
<     def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = "1",return_json:bool=False, temperature: float = 0.0) -> dict:
---
>     def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = "1",return_json:bool=False) -> dict:
1520a1408,1409
>             # Ignore assistant placeholders that only carry the internal
>             # ``tool_call`` and have no visible content.
1529c1418
<         next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
---
>         next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model)
1576a1466
>       
2569c2459,2460
< 
---
>             # Special handling for list[str] / List[str] annotations so that the
>             # generated JSON schema correctly represents an array of strings.
2800c2691,2694
<  
---
>             
>             # Add maintainability index analysis if needed
>             # Add halstead metrics analysis if needed
>             
3174c3068,3070
< 
---
>         # if type(solutions) is not list or len(solutions) < 2:
>         #     raise ToolManager.Error(ToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, f"Error: solutions must be a list with length at least 2.")
>         
3622a3519
>             # Produce a clean, parseable patch (no colors; standard unified diff).
3627a3525
>             # Log stderr separately so it never pollutes the patch.
3667a3566,3567
>         # Parse the file's AST to collect import statements
>         
3674c3574,3575
< 
---
>                 # Use the module specified in 'from x import y' if available;
>                 # otherwise fall back to the imported name from plain 'import x'
3682a3584,3585
> 
>                
3683a3587
>                 # Skip relative imports ("from . import foo") which have level > 0
3934d3837
< 
4273c4176,4177
< 
---
>             #patch=get_final_git_patch()
>             #qa_response=QA.fetch_qa_response(investigation_summary,patch)
4289c4193,4194
< 
---
>         #patch=get_final_git_patch()
>         #qa_response=QA.fetch_qa_response(investigation_summary,patch)
4453c4358,4359
< 
---
>         # Add exclusions for files in the global exclude_file_path set
>         # extra_option = " --exclude='*test*.py' --exclude='*tests.py' --exclude='test_*.py' --exclude='*_test.py' --exclude-dir='tests' --exclude-dir='testing' --include='*.py'"
4661c4567,4568
< 
---
>         # Remove passing tests from global test_func_code (only keep failing ones)
>         # global test_func_code
4662a4570,4572
>         # test_func_code = [test_func_code[i] for i in failed_test_indices]
>         
>         # Categorize failed tests by type for better guidance
4966c4876,4878
< 
---
>             
>             # Parse the test summary to distinguish actual failures from expected failures
>             # Count FAILED lines directly from short test summary
4984c4896,4898
< 
---
>                 
>                 # Extract all "number word" patterns from the summary line
>                 # This handles any order and missing sections
5103c5017,5018
< 
---
>                         # Smart truncation: keep the beginning (test name, error) and end (actual failure)
>                         # Split the failure to preserve the most important parts
5105c5020,5022
< 
---
>                         
>                         # Always keep first 20 lines (test name, setup, initial context)
>                         # And last 15 lines (actual error, assertion failure)
5248d5164
< 
5263d5178
< 
5933c5848,5849
< 
---
>         
>         # Count functions that start with 'test_'
6076c5992,5993
< 
---
>     
>     # If all retries failed, proceed with unittest
6090,6091c6007,6008
< 
< 
---
>     # setting environment to include current working directory and lib directory
>     
6108c6025,6026
< 
---
>     
>     # Preprocessing step: search in all files
6194a6113
>     # return {"patch": patch_text, "test_func_names": list(test_func_names), "logs": logs, "test_patch_find_messages": [], "patch_find_messages": [], "elapsed_time": time.time() - workflow_start_time}
6202a6122,6124
>     # if test_mode:
>         # DEFAULT_PROXY_URL = "http://localhost:8001"
> 
6224a6147,6150
>     
>     # Reset temperature manager for fresh workflow
>     Network._temperature_manager.reset()
>     
6229a6156
>             # "get_file_content",
6232a6160
>             # "save_relevant_test",
6242a6171,6172
>     #QA.SYSTEM_PROMPT=QA.SYSTEM_PROMPT.format(problem_statement=problem_statement)
>     
6269,6270c6199
<             temp = 0.7 if cot.is_thought_repeated() else 0.0
<             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id, temperature=temp)
---
>             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id)
6271a6201,6207
>             
>             # Reset temperature on successful COT inference  
>             prev_temp = Network._temperature_manager.get_current_temp()
>             new_temp = Network._temperature_manager.on_success()
>             if prev_temp != new_temp:
>                 logger.info(f"[TEST_PATCH_FIND] COT success detected, resetting temperature from {prev_temp} to {new_temp}")
>                 logs.append(f"[TEST_PATCH_FIND] COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6276a6213,6218
>             
>             # Adjust temperature on COT failure
>             new_temp = Network._temperature_manager.on_failure()
>             logger.info(f"[TEST_PATCH_FIND] COT failure detected, adjusting temperature to {new_temp}")
>             logs.append(f"[TEST_PATCH_FIND] COT failure detected, adjusting temperature to {new_temp}\n\n")
>             
6333,6508d6274
< 
< 
< def execute_fix_workflow_v0(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", test_func_codes: List[tuple[str, str, str]] = None, test_file_paths: List[str] = None) -> tuple[str, List[str], List[str]]:
<     global run_id
<     run_id=run_id_1
<     cot=COT(latest_observations_to_keep=1000)
<     
<     # Extract test file paths from test_func_codes if not provided
<     if test_file_paths is None and test_func_codes:
<         test_file_paths = []
<         for test_func_code in test_func_codes:
<             # Extract file path from the test function code
<             if "```" in test_func_code:
<                 file_path = test_func_code.split("```")[1].split("\n")[0]
<                 if file_path and file_path not in test_file_paths:
<                     test_file_paths.append(file_path)
<     
<     tool_manager=ToolManager(
<         available_tools=[
<             "search_in_all_files_content_v2",
<             "analyze_test_coverage",
<             "analyze_dependencies",
<             "detect_code_smells",
<             "analyze_git_history",
<             "get_code_quality_metrics",
<             "validate_solution",
<             "propose_solutions",
<             "compare_solutions",
<             "apply_code_edit",
<             "get_approval_for_solution",
<             "run_repo_tests",  # Added for validation
<             "start_over",
<             "finish",
<             # 🚀 NEW: Enhanced Accuracy Tools
<             "execute_self_consistency_analysis",
<             "execute_intelligent_search", 
<             "enhanced_problem_analysis",
<             "parallel_codebase_analysis",
<             "parallel_test_discovery",
<             "parallel_file_operations"
<         ],
<         test_files=test_file_paths or []
<     )
<     logger.info(f"Starting main agent execution...")
<     system_prompt = FIX_SYSTEM_PROMPT_TEMPLATE_V0.format(tools_docs=ToolManager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
<     instance_prompt = INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement, test_func_codes="\n\n".join(test_func_codes))
< 
<     logger.info(f"instance_prompt: {instance_prompt}")
< 
<     start_time = time.time()
<     logs: List[str] = []
<     logs.append(f"cwd: {os.getcwd()}")
<     logger.info(f"Starting workflow execution with {MAX_STEPS} max steps: timeout: {timeout} seconds : run_id: {run_id}")
< 
<     for step in range(MAX_STEPS):
<         logger.info(f"Execution step {step + 1}/{MAX_STEPS}")
<         
<         if time.time() - start_time > timeout:
<             cot.add_action(COT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
<             break
< 
<         messages: List[Dict[str, Any]] = [
<                 {"role": "system", "content": system_prompt},
<                 {"role": "user", "content": instance_prompt},
<             ]
<         
<         messages.extend(cot.to_str())
<         messages.append({"role": "system", "content": STOP_INSTRUCTION})
< 
<         if cot.is_thought_repeated():
<             logger.info(f"[MAIN] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
<             last_thought = cot.thoughts[-1]
<             messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
<     
<         try:
<             temp = 0.7 if cot.is_thought_repeated() else 0.0
<             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id, temperature=temp)
<             logs.append(f"next_thought: {next_thought}\n\nnext_tool_name: {next_tool_name}\n\nnext_tool_args: {next_tool_args}\n\n")
<         except Exception as e:
<             import traceback  # Ensure traceback is accessible
<             error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
<             logs.append(f"Inference error: {error_msg}\n\n")
<             logger.error(f"Inference error: {error_msg}")
<             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
<             break
<         
<         logger.info(f"About to execute operation: {next_tool_name}")
<        
<         try:
<             logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
<             if '"' in next_tool_name or "'" in next_tool_name:
<                 next_tool_name=next_tool_name.replace('"','')
<                 next_tool_name=next_tool_name.replace("'","")
<                 
<             next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
<             logs.append(f"next_observation: {next_observation}\n\n")
<             logger.info(f"next_observation: {next_observation}")
<             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
<         except ToolManager.Error as e:
<             import traceback  # Ensure traceback is accessible
<             error_msg=f"observation: {e.message}"
<             logs.append(f"Tool error: {error_msg}\n\n")
<             logger.error(f"Tool error: {error_msg}")
<             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
<             continue
<         except Exception as e:
<             import traceback  # Ensure traceback is accessible
<             error_traceback=traceback.format_exc()
<             if isinstance(e,TypeError):
<                 error_msg=f"observation: {str(e)}"
<             else:
<                 error_msg=f"observation: {repr(e)} {error_traceback}"
<             logs.append(f"Tool error: {error_msg}\n\n")
<             logger.error(f"Tool error: {error_msg}")
<             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
<             continue
<         
<         if next_tool_name == "finish":
<             logs.append(f"Workflow called finish operation\n\n")
<             logger.info('[CRITICAL] Workflow called finish operation')
<             break
<         logs.append(f"Completed step {step + 1}, continuing to next step\n\n")
<         print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
<     else:
<         # This happens if we exit the loop without breaking (reached MAX_STEPS)
<         cot.add_action(COT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
<         logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({MAX_STEPS})")
<     
<     logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
<     logger.info(f"[CRITICAL] About to generate final patch...")
<     patch = tool_manager.get_final_git_patch()
<     logger.info(f"Final Patch Generated..: Length: {len(patch)}")
<     logger.info(f"Final Patch: {patch}")
<     logs.append(f"Final Patch: {patch}\n\n")
<     
< 
<     return patch, logs
< 
< class CircuitBreaker:
<     """Circuit breaker pattern for fault tolerance"""
<     
<     def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
<         self.failure_threshold = failure_threshold
<         self.recovery_timeout = recovery_timeout
<         self.failure_count = 0
<         self.last_failure_time = 0
<         self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
<     
<     def call(self, func: callable, *args, **kwargs) -> Any:
<         """Execute function with circuit breaker protection"""
<         if self.state == 'OPEN':
<             if time.time() - self.last_failure_time > self.recovery_timeout:
<                 self.state = 'HALF_OPEN'
<             else:
<                 raise Exception("Circuit breaker is OPEN")
<         
<         try:
<             result = func(*args, **kwargs)
<             self._on_success()
<             return result
<         except Exception as e:
<             self._on_failure()
<             raise e
<     
<     def _on_success(self):
<         """Handle successful execution"""
<         self.failure_count = 0
<         self.state = 'CLOSED'
<     
<     def _on_failure(self):
<         """Handle failed execution"""
<         self.failure_count += 1
<         self.last_failure_time = time.time()
<         
<         if self.failure_count >= self.failure_threshold:
<             self.state = 'OPEN'
6513a6280,6282
>     # Reset temperature manager for fresh workflow
>     Network._temperature_manager.reset()
>     
6540c6309
< 
---
>     
6571,6572c6340
<             temp = 0.7 if cot.is_thought_repeated() else 0.0
<             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id, temperature=temp)
---
>             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id)
6577a6346,6352
>             
>             # Reset temperature on successful COT inference  
>             prev_temp = Network._temperature_manager.get_current_temp()
>             new_temp = Network._temperature_manager.on_success()
>             if prev_temp != new_temp:
>                 logger.info(f"COT success detected, resetting temperature from {prev_temp} to {new_temp}")
>                 logs.append(f"COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6582a6358,6363
>             
>             # Adjust temperature on COT failure
>             new_temp = Network._temperature_manager.on_failure()
>             logger.info(f"COT failure detected, adjusting temperature to {new_temp}")
>             logs.append(f"COT failure detected, adjusting temperature to {new_temp}\n\n")
>             
6656a6438,6441
>     
>     # Reset temperature manager for fresh workflow
>     EnhancedNetwork._temperature_manager.reset()
>     
6659a6445
>     logger.info(f"[{log_prefix}] Temperature manager reset to {EnhancedNetwork._temperature_manager.get_current_temp()}")
6714a6501,6503
>         # if last_try_summarization:
>         #     instance_prompt += f
> 
6763,6764c6552
<             eff_temp = 0.7 if cot.is_thought_repeated() else 0.0
<             next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(messages, model=current_model, run_id=run_id, temperature=eff_temp)
---
>             next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(messages, model=current_model, run_id=run_id)
6770a6559,6565
>             
>             # Reset temperature on successful COT inference
>             prev_temp = EnhancedNetwork._temperature_manager.get_current_temp()
>             new_temp = EnhancedNetwork._temperature_manager.on_success()
>             if prev_temp != new_temp:
>                 logger.info(f"[{log_prefix}] COT success detected, resetting temperature from {prev_temp} to {new_temp}")
>                 logs.append(f"[{log_prefix}] COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6775a6571,6576
>             
>             # Adjust temperature on COT failure
>             new_temp = EnhancedNetwork._temperature_manager.on_failure()
>             logger.info(f"[{log_prefix}] COT failure detected, adjusting temperature to {new_temp}")
>             logs.append(f"[{log_prefix}] COT failure detected, adjusting temperature to {new_temp}\n\n")
>             
6878c6679,6680
< 
---
>         
>         # Check for finish condition
6928c6730,6731
< 
---
>         
>         # Build instance prompt
6985c6788,6789
< 
---
>     
>     # Build instance prompt - choose template based on whether problem statement contains Python code
7007d6810
<         # upgrade_model_time=700,
7020a6824
> 
7045a6850,6851
>     
>     # Extract hints from problem statement
7110c6916
<       
---
>             # separate file path and function name            
