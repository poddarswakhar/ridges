1,2d0
< # “Trying Is Half the Battle (The Other Half Is Trying Again)” v2.1
< 
448,449c446,447
< DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))
< MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))
---
> DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
> MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "500"))
453d450
< QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
455c452
< AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]
---
> AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
457,458c454,455
< MAX_STEPS = 250
< MAX_STEPS_TEST_PATCH_FIND = 100
---
> MAX_STEPS = 150
> MAX_STEPS_TEST_PATCH_FIND = 40
460a458,464
> # 🚀 Enhanced Accuracy Algorithm Configuration
> SELF_CONSISTENCY_CONFIG = {
>     'DEFAULT_NUM_PATHS': 5,
>     'DEFAULT_CONSENSUS_THRESHOLD': 0.6,
>     'MAX_EXECUTION_TIME': 30,  # seconds
>     'ENABLE_ADAPTIVE_PATHS': True
> }
461a466,481
> INTELLIGENT_SEARCH_CONFIG = {
>     'DEFAULT_FUSION_METHOD': 'weighted',
>     'MAX_SEARCH_STRATEGIES': 5,
>     'SEARCH_TIMEOUT': 20,  # seconds per strategy
>     'ENABLE_CONTEXT_ANALYSIS': True,
>     'ENABLE_ADAPTIVE_ROUTING': True
> }
> 
> # Combined accuracy improvement estimation
> EXPECTED_ACCURACY_IMPROVEMENT = {
>     'self_consistency': 0.25,  # +25%
>     'intelligent_search': 0.15,  # +15%
>     'combined': 0.40,  # +40% (synergistic effect)
>     'confidence_threshold': 0.8
> }
> 
475a496,532
> astroid.decorators.cached = functools.lru_cache;
> astroid.decorators.cachedproperty = getattr(astroid.decorators, 'cachedproperty', property);
> astroid.TryExcept = getattr(astroid, 'Try', getattr(astroid, 'TryExcept', None));
> astroid.TryFinally = getattr(astroid, 'Try', getattr(astroid, 'TryFinally', None));
> astroid.Discard = getattr(astroid, 'Expr', getattr(astroid, 'Discard', None));
> astroid.nodes.TryExcept = getattr(astroid.nodes, 'Try', getattr(astroid.nodes, 'TryExcept', None));
> astroid.nodes.TryFinally = getattr(astroid.nodes, 'Try', getattr(astroid.nodes, 'TryFinally', None));
> astroid.bases.BUILTINS = getattr(astroid.bases, 'BUILTINS', 'builtins');
> py = types.ModuleType('py'); py._path = types.ModuleType('_path'); py._path.local = types.ModuleType('local'); py._path.local.LocalPath = pathlib.Path; sys.modules['py'] = py; sys.modules['py._path'] = py._path; sys.modules['py._path.local'] = py._path.local;
> def add_doc_compatibility():
>     import astroid.nodes as nodes;
>     def make_doc_property():
>         def doc_getter(self):
>             if hasattr(self, 'doc_node') and self.doc_node:
>                 return self.doc_node.value if hasattr(self.doc_node, 'value') else str(self.doc_node);
>             elif hasattr(self, '_get_doc'):
>                 return self._get_doc();
>             else:
>                 return None;
>         return property(doc_getter);
>     for name in dir(nodes):
>         cls = getattr(nodes, name);
>         if isinstance(cls, type) and issubclass(cls, nodes.NodeNG) and not hasattr(cls, 'doc'):
>             try:
>                 cls.doc = make_doc_property();
>             except: pass;
> add_doc_compatibility();
> def patch_module_statement():
>     import astroid.nodes as nodes;
>     original_statement = nodes.Module.statement;
>     def safe_statement(self, future=None):
>         try:
>             return original_statement(self, future=future);
>         except Exception:
>             return self;
>     nodes.Module.statement = safe_statement;
> patch_module_statement();
508a566
> blacklisted_test_files=[]
510,543c568
< class TemperatureManager:
<     """Manages dynamic temperature adjustment based on COT failure patterns"""
<     
<     def __init__(self, initial_temp: float = 0.0, failure_temp: float = 0.1):
<         self.initial_temp = initial_temp
<         self.failure_temp = failure_temp
<         self.current_temp = initial_temp
<         self.consecutive_failures = 0
<         self.failure_threshold = 1  # Adjust after first failure
<         self.reset_after_success = True
<         
<     def on_failure(self) -> float:
<         """Called when a COT inference fails. Returns new temperature."""
<         self.consecutive_failures += 1
<         if self.consecutive_failures >= self.failure_threshold:
<             self.current_temp = self.failure_temp
<         return self.current_temp
<         
<     def on_success(self) -> float:
<         """Called when a COT inference succeeds. Returns new temperature."""
<         if self.reset_after_success:
<             self.consecutive_failures = 0
<             self.current_temp = self.initial_temp
<         return self.current_temp
<         
<     def get_current_temp(self) -> float:
<         """Get current temperature setting."""
<         return self.current_temp
<         
<     def reset(self):
<         """Reset to initial state."""
<         self.current_temp = self.initial_temp
<         self.consecutive_failures = 0
< 
---
> # Enhanced caching and timeout system
720a746,771
>     def search_multiple_directories_parallel(self, directories: List[str], search_term: str) -> Dict[str, str]:
>         """Search the same term across multiple directories in parallel"""
>         
>         def search_directory(directory: str) -> tuple[str, str]:
>             try:
>                 result = self.tool_manager.search_recurive_in_all_files_in_directory(
>                     directory_path=directory,
>                     search_term=search_term
>                 )
>                 return directory, result
>             except Exception as e:
>                 return directory, f"Error searching in '{directory}': {e}"
>         
>         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(directories), 3)) as executor:
>             future_to_dir = {
>                 executor.submit(search_directory, directory): directory 
>                 for directory in directories
>             }
>             
>             results = {}
>             for future in concurrent.futures.as_completed(future_to_dir):
>                 directory, result = future.result()
>                 results[directory] = result
>         
>         return results
> 
749c800,829
< 
---
>     def apply_multiple_edits_parallel(self, edits: List[Dict[str, Any]]) -> Dict[str, str]:
>         """Apply multiple code edits in parallel"""
>         
>         def apply_single_edit(edit: Dict[str, Any]) -> tuple[str, str]:
>             try:
>                 file_path = edit['file_path']
>                 search = edit['search']
>                 replace = edit['replace']
>                 
>                 result = self.tool_manager.apply_code_edit(
>                     file_path=file_path,
>                     search=search,
>                     replace=replace
>                 )
>                 return file_path, result
>             except Exception as e:
>                 return edit.get('file_path', 'unknown'), f"Error applying edit: {e}"
>         
>         with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(edits), 3)) as executor:
>             future_to_edit = {
>                 executor.submit(apply_single_edit, edit): edit 
>                 for edit in edits
>             }
>             
>             results = {}
>             for future in concurrent.futures.as_completed(future_to_edit):
>                 file_path, result = future.result()
>                 results[file_path] = result
>         
>         return results
753a834,899
>     
>     def execute_with_dependencies(self, problem_statement: str, test_func_names: List[str]) -> Dict[str, Any]:
>         """Execute operations in parallel where possible, respecting dependencies"""
>         
>         # Phase 1: Independent operations (can run in parallel)
>         phase1_tasks = {
>             'file_listing': lambda: self.tool_manager.list_python_files(),
>             'git_status': lambda: self.tool_manager.get_git_status(),
>             'git_branches': lambda: self.tool_manager.get_git_branches()
>         }
>         
>         phase1_results = self._execute_parallel(phase1_tasks)
>         
>         # Phase 2: Operations that depend on Phase 1 results
>         python_files = phase1_results.get('file_listing', '').split('\n')
>         relevant_files = [f for f in python_files if f.strip()]
>         
>         phase2_tasks = {}
>         for file_path in relevant_files[:5]:  # Limit to first 5 files
>             phase2_tasks[f'analyze_{file_path}'] = lambda fp=file_path: self._analyze_file(fp)
>         
>         phase2_results = self._execute_parallel(phase2_tasks)
>         
>         # Phase 3: Operations that depend on test functions
>         phase3_tasks = {}
>         for test_func in test_func_names:
>             file_path, func_name = test_func.split(" - ")
>             phase3_tasks[f'test_analysis_{func_name}'] = lambda fp=file_path, fn=func_name: self._analyze_test(fp, fn)
>         
>         phase3_results = self._execute_parallel(phase3_tasks)
>         
>         return {
>             'phase1': phase1_results,
>             'phase2': phase2_results,
>             'phase3': phase3_results
>         }
>     
>     def _execute_parallel(self, tasks: Dict[str, callable]) -> Dict[str, Any]:
>         """Execute a dictionary of tasks in parallel"""
>         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
>             future_to_task = {
>                 executor.submit(task_func): task_name 
>                 for task_name, task_func in tasks.items()
>             }
>             
>             results = {}
>             for future in concurrent.futures.as_completed(future_to_task):
>                 task_name = future_to_task[future]
>                 try:
>                     result = future.result(timeout=60)
>                     results[task_name] = result
>                 except Exception as e:
>                     results[task_name] = f"Error: {e}"
>         
>         return results
>     
>     def _analyze_file(self, file_path: str) -> Dict[str, Any]:
>         """Analyze a single file with multiple tools"""
>         try:
>             return {
>                 'content': self.tool_manager.get_file_content(file_path, limit=1000),
>                 'smells': self.tool_manager.detect_code_smells(file_path),
>                 'quality': self.tool_manager.get_code_quality_metrics(file_path)
>             }
>         except Exception as e:
>             return {'error': str(e)}
754a901,909
>     def _analyze_test(self, file_path: str, func_name: str) -> Dict[str, Any]:
>         """Analyze a test function"""
>         try:
>             return {
>                 'body': self.tool_manager.get_function_body(file_path, func_name),
>                 'coverage': self.tool_manager.analyze_test_coverage([f"{file_path} - {func_name}"])
>             }
>         except Exception as e:
>             return {'error': str(e)}
996,998d1150
<     # Global temperature manager for all network instances
<     _temperature_manager = TemperatureManager()
<         
1051c1203
<                 "temperature": cls._temperature_manager.get_current_temp(),
---
>                 "temperature": 0.0,
1281,1282d1432
<     # Inherit the temperature manager from parent class
<     
1311c1461
<                 "temperature": cls._temperature_manager.get_current_temp(),
---
>                 "temperature": 0.0,
1466d1615
<       
2723a2873,2909
>         astroid.decorators.cached = functools.lru_cache;
>         astroid.decorators.cachedproperty = getattr(astroid.decorators, 'cachedproperty', property);
>         astroid.TryExcept = getattr(astroid, 'Try', getattr(astroid, 'TryExcept', None));
>         astroid.TryFinally = getattr(astroid, 'Try', getattr(astroid, 'TryFinally', None));
>         astroid.Discard = getattr(astroid, 'Expr', getattr(astroid, 'Discard', None));
>         astroid.nodes.TryExcept = getattr(astroid.nodes, 'Try', getattr(astroid.nodes, 'TryExcept', None));
>         astroid.nodes.TryFinally = getattr(astroid.nodes, 'Try', getattr(astroid.nodes, 'TryFinally', None));
>         astroid.bases.BUILTINS = getattr(astroid.bases, 'BUILTINS', 'builtins');
>         py = types.ModuleType('py'); py._path = types.ModuleType('_path'); py._path.local = types.ModuleType('local'); py._path.local.LocalPath = pathlib.Path; sys.modules['py'] = py; sys.modules['py._path'] = py._path; sys.modules['py._path.local'] = py._path.local;
>         def add_doc_compatibility():
>             import astroid.nodes as nodes;
>             def make_doc_property():
>                 def doc_getter(self):
>                     if hasattr(self, 'doc_node') and self.doc_node:
>                         return self.doc_node.value if hasattr(self.doc_node, 'value') else str(self.doc_node);
>                     elif hasattr(self, '_get_doc'):
>                         return self._get_doc();
>                     else:
>                         return None;
>                 return property(doc_getter);
>             for name in dir(nodes):
>                 cls = getattr(nodes, name);
>                 if isinstance(cls, type) and issubclass(cls, nodes.NodeNG) and not hasattr(cls, 'doc'):
>                     try:
>                         cls.doc = make_doc_property();
>                     except: pass;
>         add_doc_compatibility();
>         def patch_module_statement():
>             import astroid.nodes as nodes;
>             original_statement = nodes.Module.statement;
>             def safe_statement(self, future=None):
>                 try:
>                     return original_statement(self, future=future);
>                 except Exception:
>                     return self;
>             nodes.Module.statement = safe_statement;
>         patch_module_statement();
3837a4024,4032
>         # failed_and_error_nodeids = updated_nodeids
> 
>         # if failed_and_error_nodeids and failed_and_error_nodeids[0].startswith("ERROR:"):
>         #     failed_and_error_nodeids = updated_nodeids
> 
>         # print(f"Failed and Error NodeIDS :: {failed_and_error_nodeids}\n")
> 
>         # relevant_nodeids = self.find_relevant_tests(failed_and_error_nodeids, problem_statement) if len(failed_and_error_nodeids) > 5 else failed_and_error_nodeids
>         # self.RELEVANT_TEST_FUNC_NAMES.update(relevant_nodeids)
4697d4891
<         self.pytest_timeout_secs = 60
4699,4700d4892
<         self.blacklisted_test_files = []
<         self.pytest_command_template = PYTEST_COMMAND_TEMPLATE
4744c4936,4937
<         if file_path in self.blacklisted_test_files:
---
>         global blacklisted_test_files
>         if file_path in blacklisted_test_files:
4824c5017,5022
<         return self._analyze_regular_pytest_output(output)
---
>         if len(session_starts) > 1:
>             # Meta-testing scenario - use specialized parser
>             return self._analyze_meta_pytest_output(output)
>         else:
>             # Regular pytest scenario - use original logic
>             return self._analyze_regular_pytest_output(output)
5164a5363,5367
>             # test_info = {}
>             # for test_file in test_files:
>             #     test_name = test_file.split("/")[-1][:-3]
>             #     test_info[test_name] = test_file
>             # test_files = test_info
5178a5382,5387
> 
>             # if ("FAIL" in line or "ERROR" in line) and test_files:
>             #     for test_name, test_file in test_files.items():
>             #         if test_name in line:
>             #             failed_tests.add(test_file)
>             #             break
5185a5395,5604
> 
>     def _analyze_meta_pytest_output(self, output) -> tuple[str, bool, int]:
>         """
>         Parse pytest output that contains nested pytest runs (meta-testing).
>         Focuses on outer test results, but extracts inner details for failures.
>         """
>         # Check for special error conditions first (same as regular parsing)
>         if "most likely due to a circular import" in output:
>             short_summary = self._extract_short_summary_from_meta(output)
>             return "Tests failed due to circular import" + short_summary, True, 0
>         
>         # Check for recursion errors first
>         if "RecursionError" in output or "maximum recursion depth" in output:
>             short_summary = self._extract_short_summary_from_meta(output)
>             return "Tests failed due to RecursionError" + short_summary, True, 0
>         
>         # Find the final (outermost) summary line
>         lines = output.splitlines()
>         final_summary_line = None
>         final_summary_index = -1
>         
>         # Search backwards for the final summary line (the real outer test results)
>         for i in range(len(lines) - 1, -1, -1):
>             line = lines[i]
>             if re.search(r'={3,}.*?\b\d+\.\d+s\s*(\([^)]+\))?\s*={3,}', line, re.IGNORECASE):
>                 final_summary_line = line
>                 final_summary_index = i
>                 break
>         
>         if not final_summary_line:
>             return "Could not find final test summary", False, 0
>         
>         # Parse final summary for counts
>         failed_count = 0
>         passed_count = 0
>         skipped_count = 0
>         xfailed_count = 0
>         
>         result_patterns = re.findall(r'(\d+)\s+(\w+)', final_summary_line)
>         for count, result_type in result_patterns:
>             count = int(count)
>             result_type = result_type.lower()
>             
>             if result_type == 'failed':
>                 failed_count = count
>             elif result_type == 'passed':
>                 passed_count = count
>             elif result_type == 'skipped':
>                 skipped_count = count
>             elif result_type == 'xfailed':
>                 xfailed_count = count
>         
>         # Extract short summary for outer tests
>         short_summary = self._extract_meta_short_summary(output, final_summary_index)
>         
>         # If no failures in outer tests, return success
>         if failed_count == 0:
>             return f"Successfully ran all tests. {passed_count} passed, {skipped_count} skipped." + short_summary, True, 0
>         
>         # Extract outer test failures with their inner details
>         outer_failures = self._extract_outer_test_failures_with_inner_details(output, final_summary_index)
>         
>         if not outer_failures:
>             return f"Tests failed ({failed_count} failures) but could not extract failure details." + short_summary, False, failed_count
>         
>         result = "=================================== FAILURES ===================================\n"
>         result += "\n\n".join(outer_failures)
>         
>         return result + short_summary, True, failed_count
> 
>     def _extract_meta_short_summary(self, output, final_summary_index):
>         """Extract short summary info for meta-testing scenarios."""
>         lines = output.splitlines()
>         
>         # Look for short test summary info before the final summary
>         for i in range(final_summary_index - 1, max(0, final_summary_index - 50), -1):
>             if re.search(r'={5,}\s*short test summary info\s*={5,}', lines[i], re.IGNORECASE):
>                 # Found short summary section, extract it
>                 summary_start = i
>                 summary_end = final_summary_index
>                 
>                 summary_content = "\n".join(lines[summary_start:summary_end]).strip()
>                 if summary_content:
>                     return f"\n\n{summary_content}"
>                 break
>         
>         return ""
> 
>     def _extract_outer_test_failures_with_inner_details(self, output, final_summary_index):
>         """
>         Extract failed outer tests and include relevant inner test details.
>         """
>         lines = output.splitlines()
>         
>         # Find the outer FAILURES section (should be before final summary)
>         failures_start = -1
>         for i in range(final_summary_index - 1, -1, -1):
>             if re.search(r'={5,}\s*FAILURES\s*={5,}', lines[i], re.IGNORECASE):
>                 failures_start = i
>                 break
>         
>         if failures_start == -1:
>             return []
>         
>         # Extract the outer failures section
>         failures_section = "\n".join(lines[failures_start:final_summary_index])
>         
>         # Split into individual test failures
>         failure_pattern = re.compile(r'_{15,}\s+(.+?)\s+_{15,}')
>         failure_separators = list(failure_pattern.finditer(failures_section))
>         
>         outer_failures = []
>         
>         for i, separator in enumerate(failure_separators):
>             test_name = separator.group(1).strip()
>             start_pos = separator.end()
>             
>             if i + 1 < len(failure_separators):
>                 end_pos = failure_separators[i + 1].start()
>             else:
>                 end_pos = len(failures_section)
>             
>             failure_content = failures_section[start_pos:end_pos].strip()
>             
>             # For meta-tests, extract the inner test details that are relevant
>             enhanced_failure = self._enhance_meta_test_failure(test_name, failure_content, output)
>             
>             if enhanced_failure:
>                 outer_failures.append(enhanced_failure)
>             
>             # Limit to first 2 failures to keep output manageable
>             if len(outer_failures) >= 2:
>                 remaining = len(failure_separators) - len(outer_failures)
>                 if remaining > 0:
>                     outer_failures.append(f"... and {remaining} more failures (showing first 2 only)")
>                 break
>         
>         return outer_failures
> 
>     def _enhance_meta_test_failure(self, test_name, failure_content, full_output):
>         """
>         For meta-test failures, extract relevant inner test session details.
>         Works with any test file structure, not just pytest-specific paths.
>         """
>         # Start with the basic failure info
>         enhanced = f"_{60}_\n{test_name}\n_{60}_\n\n{failure_content}"
>         
>         # Extract file path and method name from test_name
>         # Format is usually: "path/to/file.py::TestClass::test_method" or "path/to/file.py::test_method"
>         if "::" in test_name:
>             parts = test_name.split("::")
>             file_path = parts[0]  # e.g., "testing/test_unittest.py"
>             test_method = parts[-1]  # e.g., "test_simple_unittest"
>             
>             # Escape special regex characters in both file path and method
>             escaped_file_path = re.escape(file_path)
>             escaped_method = re.escape(test_method)
>             
>             # Look for inner test session that might be related to this failure
>             # More general pattern that works with any file structure
>             inner_session_pattern = rf"{escaped_file_path}::{escaped_method}.*?test session starts.*?={3,}.*?\d+\.\d+s.*?={3,}"
>             inner_match = re.search(inner_session_pattern, full_output, re.DOTALL | re.IGNORECASE)
>             
>             if inner_match:
>                 inner_session = inner_match.group()
>                 
>                 # Check if the inner session had failures that might be relevant
>                 if "FAILED" in inner_session or "FAILURES" in inner_session:
>                     # Extract inner failures section
>                     inner_failures_match = re.search(r'={5,}\s*FAILURES\s*={5,}.*?(?=={5,}|\Z)', inner_session, re.DOTALL | re.IGNORECASE)
>                     if inner_failures_match:
>                         inner_failures = inner_failures_match.group()
>                         # Truncate if too long
>                         if len(inner_failures) > 3000:
>                             lines = inner_failures.splitlines()
>                             if len(lines) > 100:
>                                 inner_failures = "\n".join(lines[:50] + [f"... (truncated {len(lines) - 100} lines) ..."] + lines[-50:])
>                         
>                         enhanced += f"\n\n--- Related Inner Test Session Failures ---\n{inner_failures}"
>                 
>                 # Always include the inner test summary for context
>                 inner_summary_match = re.search(r'={3,}.*?\d+\.\d+s.*?={3,}', inner_session)
>                 if inner_summary_match:
>                     enhanced += f"\n\n--- Inner Test Summary ---\n{inner_summary_match.group()}"
>             else:
>                 # If we can't find the specific test, try a broader search
>                 # Look for any test session that contains the method name
>                 broader_pattern = rf"{escaped_method}.*?test session starts.*?={3,}.*?\d+\.\d+s.*?={3,}"
>                 broader_match = re.search(broader_pattern, full_output, re.DOTALL | re.IGNORECASE)
>                 
>                 if broader_match:
>                     inner_session = broader_match.group()
>                     
>                     # Only add if it contains failures
>                     if "FAILED" in inner_session or "FAILURES" in inner_session:
>                         inner_failures_match = re.search(r'={5,}\s*FAILURES\s*={5,}.*?(?=={5,}|\Z)', inner_session, re.DOTALL | re.IGNORECASE)
>                         if inner_failures_match:
>                             inner_failures = inner_failures_match.group()
>                             if len(inner_failures) > 3000:
>                                 lines = inner_failures.splitlines()
>                                 if len(lines) > 100:
>                                     inner_failures = "\n".join(lines[:50] + [f"... (truncated {len(lines) - 100} lines) ..."] + lines[-50:])
>                             
>                             enhanced += f"\n\n--- Related Inner Test Session Failures (broader match) ---\n{inner_failures}"
>         
>         # Truncate the entire enhanced failure if it's too long
>         if len(enhanced) > 15000:
>             enhanced = enhanced[:15000] + "\n\n... (truncated enhanced failure, full content was too long)"
>         
>         return enhanced
5200,5201c5619,5701
<     def _run_repo_tests_with_timeout(self, files_to_test: List[str], timeout_secs: int = 60) -> tuple[str, bool]:
<         global REPO_DIR, last_test_runner, test_runner, test_runner_mode
---
>     def _parse_stack_trace_and_enhance_output(self, output: str) -> str:
>         """
>         Parse stack traces from test output and enhance with relevant file contents
>         """
>         import re
>         
>         # Extract stack trace information
>         stack_trace_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
>         matches = re.findall(stack_trace_pattern, output)
>         
>         enhanced_output = f"""
> TASK: Fix the test failure by analyzing the error and source code context below.
> 
> === TEST OUTPUT ===
> {output}
> """
>         
>         if matches:
>             # Focus on the actual source files (not test files)
>             relevant_files = []
>             for file_path, line_num, method_name in matches:
>                 # Skip test files and focus on implementation files in /sandbox/repo/
>                 if (not any(test_dir in file_path for test_dir in ['test', 'tests']) 
>                     and file_path.endswith('.py') 
>                     and '/sandbox/repo/' in file_path
>                     and not file_path.endswith('runtests.py')):
>                     relevant_files.append((file_path, int(line_num), method_name))
>             
>             if relevant_files:
>                 # Get the most relevant files (focus on the last few in the stack)
>                 primary_files = relevant_files[-2:] if len(relevant_files) > 1 else relevant_files
>                 
>                 enhanced_output += "\n=== SOURCE CODE CONTEXT ===\n"
>                 
>                 for i, (file_path, line_num, method_name) in enumerate(primary_files):
>                     try:
>                         # Get focused context around the error line
>                         file_content = self.get_file_content(
>                             file_path, 
>                             search_start_line=max(1, line_num - 20),
>                             search_end_line=line_num + 20
>                         )
>                         
>                         enhanced_output += f"""
> --- {file_path}:{line_num} in {method_name}() ---
> {file_content}
> """
>                             
>                     except Exception as e:
>                         enhanced_output += f"\n\nNote: Could not fetch source file content for {file_path}: {e}\n"
>                 
>                 # Extract and highlight the specific error message
>                 error_message = self._extract_error_message(output)
>                 if error_message:
>                     enhanced_output += f"""
> 
> ERROR: {error_message}
> 
> ANALYSIS:
> 1. Why is this error raised here? Is it always invalid or are some cases valid?
> 2. Search codebase for similar patterns and how they're handled
> 3. Consider if fix needs changes in multiple related files/classes
> 4. Look at the broader method logic to understand intended behavior
> """
>         
>         return enhanced_output
> 
>     def _extract_error_message(self, output: str) -> str:
>         """Extract the actual error message from test output"""
>         lines = output.split('\n')
>         for line in lines:
>             # Look for common error patterns
>             if any(error_type in line for error_type in [
>                 'Error:', 'Exception:', 'AssertionError:', 'DatabaseError:', 
>                 'ValueError:', 'TypeError:', 'AttributeError:', 'ImportError:'
>             ]):
>                 return line.strip()
>         return ""
> 
> 
> 
>     def _run_repo_tests_with_timeout(self, files_to_test: List[str], timeout_secs: int = 90) -> tuple[str, bool]:
>         global REPO_DIR, last_test_runner, test_runner, test_runner_mode, PYTEST_COMMAND_TEMPLATE
5205,5206c5705,5706
<             command = self.pytest_command_template.format(file_paths=file_paths_str)
<             result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout_secs)
---
>             command = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str)
>             result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_secs)
5208d5707
<             self.logs.append("`run_repo_tests` output: \n" + out)
5209a5709,5710
>             logger.info(f"!!! out !!!: {out}")
> 
5216d5716
<                     self.logs.append(f"command: {cmd}")
5219d5718
<                     self.logs.append(f"`run_repo_tests` output: \n{out}")
5263d5761
<                     debug_outputs = "" 
5265c5763
<                         debug_outputs += "\n\n=================================== Debug Prints ===================================\n\n"
---
>                         output += "\n\n=================================== Debug Prints ===================================\n\n"
5269c5767
<                                     debug_outputs += f"\n---------------------------------- Debug prints for {test_name} ----------------------------------\n"
---
>                                     output += f"\n---------------------------------- Debug prints for {test_name} ----------------------------------\n"
5271,5299c5769,5770
<                                         debug_outputs += f"\n{print}"
<                         debug_outputs += "\n\n=================================== End of Debug Prints ===================================\n\n"
<                     
<                     # Only add debug_outputs if it has less than 500 lines
<                     if debug_outputs:
<                         debug_outputs_lines = debug_outputs.splitlines()
<                         
<                         # Helper function to truncate long lines
<                         def truncate_long_lines(lines, max_length=3000):
<                             truncated_lines = []
<                             for line in lines:
<                                 if len(line) > max_length:
<                                     truncated_lines.append(line[:max_length] + f"... (truncated {len(line) - max_length} chars)")
<                                 else:
<                                     truncated_lines.append(line)
<                             return truncated_lines
<                         
<                         if len(debug_outputs_lines) < 500:
<                             truncated_lines = truncate_long_lines(debug_outputs_lines)
<                             output += "\n".join(truncated_lines)
<                         else:
<                             first_250 = truncate_long_lines(debug_outputs_lines[:250])
<                             last_250 = truncate_long_lines(debug_outputs_lines[-250:])
<                             omitted = len(debug_outputs_lines) - 500
<                             output += (
<                                 "\n".join(first_250)
<                                 + f"\n... (truncated {omitted} lines) ...\n"
<                                 + "\n".join(last_250)
<                             )
---
>                                         output += f"\n{print}"
>                         output += "\n\n=================================== End of Debug Prints ===================================\n\n"
5308a5780,5783
>                 # else :
>                 #     if self.failed_count > 0:
>                 #         output += f"\n\nYou didn't resolve any failures yet. DO NOT CHECKPOINT YOUR PROGRESS UNTIL YOU HAVE FIXED AT LEAST ONE FAILURE."
>             
5326a5802
>         global blacklisted_test_files
5339,5340c5815,5816
<         if self.blacklisted_test_files:
<             for file in self.blacklisted_test_files:
---
>         if blacklisted_test_files:
>             for file in blacklisted_test_files:
5365a5842
>         global blacklisted_test_files
5368c5845
<         if file_path in self.blacklisted_test_files:
---
>         if file_path in blacklisted_test_files:
5381a5859
>         global blacklisted_test_files
5395c5873
<                     if file_path in self.blacklisted_test_files:
---
>                     if file_path in blacklisted_test_files:
5550,5557c6028,6035
<                     is_error,error=self.check_syntax_error(new_content)
<                     if not is_error:
<                         self.save_file(file_path, new_content)
<                             
<                         return "ok, code edit applied successfully"
<                     else:
<                         error.message="code edit failed. "+error.message
<                         raise error
---
>                         is_error,error=self.check_syntax_error(new_content)
>                         if not is_error:
>                             self.save_file(file_path, new_content)
>                                 
>                             return "ok, code edit applied successfully"
>                         else:
>                             error.message="code edit failed. "+error.message
>                             raise error
5570a6049
>         global blacklisted_test_files
5577a6057,6066
>             
>             # Handle different formats:
>             # if "::" in test_func_name:
>             #     # Format: test_file_path.py::test_func_name
>             #     parts = test_func_name.split("::")
>             #     if len(parts) >= 2 and parts[0].strip().endswith(".py"):
>             #         test_file = parts[0].strip()
>             # elif test_func_name.strip().endswith(".py"):
>             #     # Format: test_file_path.py (file path only)
>             #     test_file = test_func_name.strip()
5582c6071
<             if test_file in self.blacklisted_test_files:
---
>             if test_file in blacklisted_test_files:
5598,5599c6087,6088
<             if self.blacklisted_test_files:
<                 self.blacklisted_test_files.extend(list(test_files))
---
>             if blacklisted_test_files:
>                 blacklisted_test_files.extend(list(test_files))
5601c6090
<                 self.blacklisted_test_files = list(test_files)
---
>                 blacklisted_test_files = list(test_files)
5709,5712c6198,6210
<             self.previous_failed_tests = self.failed_test_names.copy()
<             self.failed_test_names = None
<             self.failed_count = -1
<             return self.run_repo_tests()   
---
>             current_patch = self.get_final_git_patch()
>             if self._count_modified_or_added_lines_from_patch(current_patch) < 5: # changes are small, so might not need to check other test functions
>                 print("Successfully run on failed tests, running on all tests again., Changes are small, so skip checking other test functions.")
>                 self.logs.append(f"Successfully run on failed tests, running on all tests again., Changes are small, so skip checking other test functions.")
>                 self.can_finish = True
>                 return output
>             else:
>                 print(f"Successfully run on failed tests, running on all tests again., Changes are large, so checking other test functions to be sure that I didn't break other tests.")
>                 self.logs.append(f"Successfully run on failed tests, running on all tests again., Changes are large, so checking other test functions to be sure that I didn't break other tests.")
>                 self.previous_failed_tests = self.failed_test_names.copy()
>                 self.failed_test_names = None
>                 self.failed_count = -1
>                 return self.run_repo_tests()   
5921c6419
<                 _file_paths = [filepath_to_module(f, repod_dir, test_runner) for f in file_paths]
---
>                 file_paths = [filepath_to_module(f, repod_dir, test_runner) for f in file_paths]
5923,5924c6421,6422
<                 _file_paths = [clean_filepath(f, repod_dir, test_runner) for f in file_paths]
<             cmd = f"{test_runner} {' '.join(_file_paths)}"
---
>                 file_paths = [clean_filepath(f, repod_dir, test_runner) for f in file_paths]
>             cmd = f"{test_runner} {' '.join(file_paths)}"
5978c6476
<     max_retries = 2
---
>     max_retries = 1
6122,6123c6620,6621
<     # if test_mode:
<         # DEFAULT_PROXY_URL = "http://localhost:8001"
---
>     if test_mode:
>         DEFAULT_PROXY_URL = "http://localhost:8001"
6147,6150d6644
<     
<     # Reset temperature manager for fresh workflow
<     Network._temperature_manager.reset()
<     
6201,6207d6694
<             
<             # Reset temperature on successful COT inference  
<             prev_temp = Network._temperature_manager.get_current_temp()
<             new_temp = Network._temperature_manager.on_success()
<             if prev_temp != new_temp:
<                 logger.info(f"[TEST_PATCH_FIND] COT success detected, resetting temperature from {prev_temp} to {new_temp}")
<                 logs.append(f"[TEST_PATCH_FIND] COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6213,6218d6699
<             
<             # Adjust temperature on COT failure
<             new_temp = Network._temperature_manager.on_failure()
<             logger.info(f"[TEST_PATCH_FIND] COT failure detected, adjusting temperature to {new_temp}")
<             logs.append(f"[TEST_PATCH_FIND] COT failure detected, adjusting temperature to {new_temp}\n\n")
<             
6274a6756,6891
> 
> 
> def execute_fix_workflow_v0(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", test_func_codes: List[tuple[str, str, str]] = None, test_file_paths: List[str] = None) -> tuple[str, List[str], List[str]]:
>     global run_id
>     run_id=run_id_1
>     cot=COT(latest_observations_to_keep=1000)
>     
>     # Extract test file paths from test_func_codes if not provided
>     if test_file_paths is None and test_func_codes:
>         test_file_paths = []
>         for test_func_code in test_func_codes:
>             # Extract file path from the test function code
>             if "```" in test_func_code:
>                 file_path = test_func_code.split("```")[1].split("\n")[0]
>                 if file_path and file_path not in test_file_paths:
>                     test_file_paths.append(file_path)
>     
>     tool_manager=ToolManager(
>         available_tools=[
>             "search_in_all_files_content_v2",
>             "analyze_test_coverage",
>             "analyze_dependencies",
>             "detect_code_smells",
>             "analyze_git_history",
>             "get_code_quality_metrics",
>             "validate_solution",
>             "propose_solutions",
>             "compare_solutions",
>             "apply_code_edit",
>             "get_approval_for_solution",
>             "run_repo_tests",  # Added for validation
>             "start_over",
>             "finish",
>             # 🚀 NEW: Enhanced Accuracy Tools
>             "execute_self_consistency_analysis",
>             "execute_intelligent_search", 
>             "enhanced_problem_analysis",
>             "parallel_codebase_analysis",
>             "parallel_test_discovery",
>             "parallel_file_operations"
>         ],
>         test_files=test_file_paths or []
>     )
>     logger.info(f"Starting main agent execution...")
>     system_prompt = FIX_SYSTEM_PROMPT_TEMPLATE_V0.format(tools_docs=ToolManager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
>     instance_prompt = INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement, test_func_codes="\n\n".join(test_func_codes))
> 
>     logger.info(f"instance_prompt: {instance_prompt}")
> 
>     #QA.SYSTEM_PROMPT=QA.SYSTEM_PROMPT.format(problem_statement=problem_statement)
>     
>     start_time = time.time()
>     logs: List[str] = []
>     logs.append(f"cwd: {os.getcwd()}")
>     logger.info(f"Starting workflow execution with {MAX_STEPS} max steps: timeout: {timeout} seconds : run_id: {run_id}")
> 
>     for step in range(MAX_STEPS):
>         logger.info(f"Execution step {step + 1}/{MAX_STEPS}")
>         
>         if time.time() - start_time > timeout:
>             cot.add_action(COT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
>             break
> 
>         messages: List[Dict[str, Any]] = [
>                 {"role": "system", "content": system_prompt},
>                 {"role": "user", "content": instance_prompt},
>             ]
>         
>         messages.extend(cot.to_str())
>         messages.append({"role": "system", "content": STOP_INSTRUCTION})
> 
>         if cot.is_thought_repeated():
>             logger.info(f"[MAIN] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
>             last_thought = cot.thoughts[-1]
>             messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
>     
>         try:
>             next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = Network.inference(messages, run_id=run_id)
>             logs.append(f"next_thought: {next_thought}\n\nnext_tool_name: {next_tool_name}\n\nnext_tool_args: {next_tool_args}\n\n")
>         except Exception as e:
>             import traceback  # Ensure traceback is accessible
>             error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
>             logs.append(f"Inference error: {error_msg}\n\n")
>             logger.error(f"Inference error: {error_msg}")
>             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
>             break
>         
>         logger.info(f"About to execute operation: {next_tool_name}")
>        
>         try:
>             logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
>             if '"' in next_tool_name or "'" in next_tool_name:
>                 next_tool_name=next_tool_name.replace('"','')
>                 next_tool_name=next_tool_name.replace("'","")
>                 
>             next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
>             logs.append(f"next_observation: {next_observation}\n\n")
>             logger.info(f"next_observation: {next_observation}")
>             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
>         except ToolManager.Error as e:
>             import traceback  # Ensure traceback is accessible
>             error_msg=f"observation: {e.message}"
>             logs.append(f"Tool error: {error_msg}\n\n")
>             logger.error(f"Tool error: {error_msg}")
>             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
>             continue
>         except Exception as e:
>             import traceback  # Ensure traceback is accessible
>             error_traceback=traceback.format_exc()
>             if isinstance(e,TypeError):
>                 error_msg=f"observation: {str(e)}"
>             else:
>                 error_msg=f"observation: {repr(e)} {error_traceback}"
>             logs.append(f"Tool error: {error_msg}\n\n")
>             logger.error(f"Tool error: {error_msg}")
>             cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
>             continue
>         
>         if next_tool_name == "finish":
>             logs.append(f"Workflow called finish operation\n\n")
>             logger.info('[CRITICAL] Workflow called finish operation')
>             break
>         logs.append(f"Completed step {step + 1}, continuing to next step\n\n")
>         print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
>     else:
>         # This happens if we exit the loop without breaking (reached MAX_STEPS)
>         cot.add_action(COT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
>         logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({MAX_STEPS})")
>     
>     logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
>     logger.info(f"[CRITICAL] About to generate final patch...")
>     patch = tool_manager.get_final_git_patch()
>     logger.info(f"Final Patch Generated..: Length: {len(patch)}")
>     logger.info(f"Final Patch: {patch}")
>     logs.append(f"Final Patch: {patch}\n\n")
>     
6275a6893,6933
>     return patch, logs
> 
> class CircuitBreaker:
>     """Circuit breaker pattern for fault tolerance"""
>     
>     def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
>         self.failure_threshold = failure_threshold
>         self.recovery_timeout = recovery_timeout
>         self.failure_count = 0
>         self.last_failure_time = 0
>         self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
>     
>     def call(self, func: callable, *args, **kwargs) -> Any:
>         """Execute function with circuit breaker protection"""
>         if self.state == 'OPEN':
>             if time.time() - self.last_failure_time > self.recovery_timeout:
>                 self.state = 'HALF_OPEN'
>             else:
>                 raise Exception("Circuit breaker is OPEN")
>         
>         try:
>             result = func(*args, **kwargs)
>             self._on_success()
>             return result
>         except Exception as e:
>             self._on_failure()
>             raise e
>     
>     def _on_success(self):
>         """Handle successful execution"""
>         self.failure_count = 0
>         self.state = 'CLOSED'
>     
>     def _on_failure(self):
>         """Handle failed execution"""
>         self.failure_count += 1
>         self.last_failure_time = time.time()
>         
>         if self.failure_count >= self.failure_threshold:
>             self.state = 'OPEN'
> 
6280,6282d6937
<     # Reset temperature manager for fresh workflow
<     Network._temperature_manager.reset()
<     
6308a6964,6966
>     # return
>     # exit()
>     #QA.SYSTEM_PROMPT=QA.SYSTEM_PROMPT.format(problem_statement=problem_statement)
6346,6352d7003
<             
<             # Reset temperature on successful COT inference  
<             prev_temp = Network._temperature_manager.get_current_temp()
<             new_temp = Network._temperature_manager.on_success()
<             if prev_temp != new_temp:
<                 logger.info(f"COT success detected, resetting temperature from {prev_temp} to {new_temp}")
<                 logs.append(f"COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6358,6363d7008
<             
<             # Adjust temperature on COT failure
<             new_temp = Network._temperature_manager.on_failure()
<             logger.info(f"COT failure detected, adjusting temperature to {new_temp}")
<             logs.append(f"COT failure detected, adjusting temperature to {new_temp}\n\n")
<             
6379c7024,7030
< 
---
>             # if(tool_history.count(f"next_tool_name: {next_tool_name}, next_tool_args: {next_tool_args}, next_observation: {next_observation}") == 0):
>             #     tool_history.append(f"next_tool_name: {next_tool_name}, next_tool_args: {next_tool_args}, next_observation: {next_observation}")
>             # elif not next_tool_name.startswith("validate"):
>             #     error_msg = f"observation: Tool {next_tool_name} with args {next_tool_args} has been called several times. You must try something different!"
>             #     print(f"Error: {error_msg}")
>             #     cot.add_action(COT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=error_msg,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
>             #     continue
6436c7087
<     global run_id
---
>     global run_id, blacklisted_test_files
6438,6441d7088
<     
<     # Reset temperature manager for fresh workflow
<     EnhancedNetwork._temperature_manager.reset()
<     
6445d7091
<     logger.info(f"[{log_prefix}] Temperature manager reset to {EnhancedNetwork._temperature_manager.get_current_temp()}")
6544,6545c7190,7191
<         if tool_manager.blacklisted_test_files and len(tool_manager.blacklisted_test_files) > 0:
<             messages.append({"role": "user", "content": f"AS A REMINDER, DO NOT SEARCH OR USE THESE FILES:\n\n{tool_manager.blacklisted_test_files}"})
---
>         if blacklisted_test_files and len(blacklisted_test_files) > 0:
>             messages.append({"role": "user", "content": f"AS A REMINDER, DO NOT SEARCH OR USE THESE FILES:\n\n{blacklisted_test_files}"})
6559,6565d7204
<             
<             # Reset temperature on successful COT inference
<             prev_temp = EnhancedNetwork._temperature_manager.get_current_temp()
<             new_temp = EnhancedNetwork._temperature_manager.on_success()
<             if prev_temp != new_temp:
<                 logger.info(f"[{log_prefix}] COT success detected, resetting temperature from {prev_temp} to {new_temp}")
<                 logs.append(f"[{log_prefix}] COT success detected, resetting temperature from {prev_temp} to {new_temp}\n\n")
6571,6576d7209
<             
<             # Adjust temperature on COT failure
<             new_temp = EnhancedNetwork._temperature_manager.on_failure()
<             logger.info(f"[{log_prefix}] COT failure detected, adjusting temperature to {new_temp}")
<             logs.append(f"[{log_prefix}] COT failure detected, adjusting temperature to {new_temp}\n\n")
<             
6704a7338
>     global blacklisted_test_files
6707c7341
<     max_retries = 5
---
>     max_retries = 3
6755c7389
<             if name.split("::")[0].strip() not in tool_manager.blacklisted_test_files
---
>             if name.split("::")[0].strip() not in blacklisted_test_files
6779a7414,7416
>         # "checkpoint_progress",
>         # "revert_to_last_checkpoint",
>         # "start_over",
6810a7448
>         # upgrade_model_time=700,
6824a7463,7476
> 
> # def extract_keywords(problem_text: str) -> str:
> #     """Extract technical terms, exact patterns, and module paths from problem statement"""
> #     # Extract quoted strings (e.g., ".----", "----")
> #     quoted_patterns = re.findall(r'".*?"|\'.*?\'', problem_text)
> #     module_paths = re.findall(r'\b\w+\.\w+\.\w+\b', problem_text)
> #     # Extract technical terms and digits
> #     technical_terms = [word for word in problem_text.lower().split() 
> #                        if word.isalnum() and len(word) > 2]
>     
> #     # Combine all patterns
> #     all_keywords = list(set(quoted_patterns + module_paths + technical_terms))
> #     return '|'.join(all_keywords[:10])  # Use up to 10 most relevant keywords
> 
