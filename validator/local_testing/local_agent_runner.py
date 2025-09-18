"""
Local agent runner that executes agent code directly in the current Python environment
without Docker containers. This provides a much faster and easier debugging experience.
"""

import sys
import json
import traceback
import importlib.util
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import os
import subprocess
import time

from validator.sandbox.schema import SwebenchProblem, EvaluationRun
from loggers.logging_utils import get_logger

logger = get_logger(__name__)

class LocalAgentRunner:
    """Runs agent code locally without Docker containers"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ridges_local_agent_"))
        
    def cleanup(self):
        """Clean up temporary directories"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
    
    async def run_agent(self, problem: SwebenchProblem, agent_file: Path) -> EvaluationRun:
        """Run the agent on a problem locally"""
        
        # Create a temporary directory for this evaluation
        eval_dir = self.temp_dir / problem.instance_id
        eval_dir.mkdir(exist_ok=True)
        
        # Set up the repository
        repo_dir = await self._setup_repository(problem, eval_dir)
        
        # Set up the agent
        agent_dir = await self._setup_agent(agent_file, eval_dir)
        
        # Create input for the agent
        input_data = {
            "problem_statement": problem.problem_statement,
            "instance_id": problem.instance_id,
            "repo_dir": str(repo_dir),
            "run_id": f"local_{int(time.time())}"
        }
        
        # Run the agent
        result = await self._execute_agent(agent_dir, input_data, repo_dir)
        
        return result
    
    async def _setup_repository(self, problem: SwebenchProblem, eval_dir: Path) -> Path:
        """Clone and set up the repository for the problem"""
        
        repo_dir = eval_dir / "repo"
        
        if self.verbose:
            print(f"   Cloning repository: {problem.repo}...")
        
        # Clone the repository
        clone_cmd = [
            "git", "clone", 
            f"https://github.com/{problem.repo}.git",
            str(repo_dir)
        ]
        
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")
        
        # Checkout the specific commit
        checkout_cmd = ["git", "checkout", problem.base_commit]
        result = subprocess.run(checkout_cmd, cwd=repo_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout commit {problem.base_commit}: {result.stderr}")
        
        if self.verbose:
            print(f"   Repository ready at {repo_dir}")
        
        return repo_dir
    
    async def _setup_agent(self, agent_file: Path, eval_dir: Path) -> Path:
        """Set up the agent code"""
        
        agent_dir = eval_dir / "agent"
        agent_dir.mkdir(exist_ok=True)
        
        # Copy agent file
        agent_dest = agent_dir / "agent.py"
        shutil.copy2(agent_file, agent_dest)
        
        if self.verbose:
            print(f"   Agent code ready at {agent_dest}")
        
        return agent_dir
    
    async def _execute_agent(self, agent_dir: Path, input_data: Dict[str, Any], repo_dir: Path) -> EvaluationRun:
        """Execute the agent code locally"""
        
        # Change to the repository directory (this is where the agent will work)
        original_cwd = os.getcwd()
        
        try:
            # Change to repo directory
            os.chdir(repo_dir)
            
            if self.verbose:
                print(f"   Running agent in directory: {repo_dir}")
            
            # Set up remote debugging if debug mode is enabled
            if os.getenv("RIDGES_DEBUG", "false").lower() == "true":
                try:
                    # Add the main Python site-packages to path to find debugpy
                    import sys
                    import site
                    for site_dir in site.getsitepackages():
                        if site_dir not in sys.path:
                            sys.path.insert(0, site_dir)
                    
                    import debugpy
                    if not debugpy.is_client_connected():
                        debugpy.listen(("localhost", 5678))
                        print("🐛 Remote debugger listening on localhost:5678")
                        print("🐛 Attach PyCharm debugger to localhost:5678")
                        if os.getenv("RIDGES_DEBUG_WAIT", "false").lower() == "true":
                            debugpy.wait_for_client()
                            print("🐛 PyCharm debugger attached!")
                        
                except ImportError as e:
                    print(f"🐛 debugpy not available: {e}, using pdb for debugging")
            
            # Add agent directory to Python path
            agent_path = str(agent_dir)
            if agent_path not in sys.path:
                sys.path.insert(0, agent_path)
            
            # Import and run the agent
            spec = importlib.util.spec_from_file_location("agent", agent_dir / "agent.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the agent has the expected function
            if not hasattr(module, 'multi_task_process'):
                raise RuntimeError("Agent module must have a 'multi_task_process' function")
            
            # Run the agent
            start_time = time.time()
            
            if self.verbose:
                print(f"   Executing agent...")
            
            # Call the agent's main function
            result = module.multi_task_process(input_data, repod_dir=str(repo_dir))
            
            execution_time = time.time() - start_time
            
            if self.verbose:
                print(f"   Agent execution completed in {execution_time:.2f}s")
            
            # Parse the result
            if isinstance(result, dict) and 'patch' in result:
                patch_content = result['patch']
                solved = result.get('solved', False)
            else:
                patch_content = str(result) if result else ""
                solved = False
            
            # Create evaluation run result
            from datetime import datetime, timezone
            evaluation_run = EvaluationRun(
                run_id=input_data.get('run_id', 'local_run'),
                evaluation_id=f"local_{input_data['instance_id']}",
                swebench_instance_id=input_data['instance_id'],
                response=patch_content,
                solved=solved,
                error=None,
                fail_to_pass_success="False",  # Would need to run tests to determine
                pass_to_pass_success="True",   # Assume tests still pass
                status="result_scored",
                started_at=datetime.now(timezone.utc)
            )
            
            return evaluation_run
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            from datetime import datetime, timezone
            return EvaluationRun(
                run_id=input_data.get('run_id', 'local_run'),
                evaluation_id=f"local_{input_data['instance_id']}",
                swebench_instance_id=input_data['instance_id'],
                response="",
                solved=False,
                error=error_msg,
                fail_to_pass_success="False",
                pass_to_pass_success="False",
                status="result_scored",
                started_at=datetime.now(timezone.utc)
            )
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
            # Remove agent path from sys.path
            if agent_path in sys.path:
                sys.path.remove(agent_path)
