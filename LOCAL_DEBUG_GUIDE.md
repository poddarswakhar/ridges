# 🚀 Local Agent Debugging Guide

This guide shows you how to run and debug your `top_agent.py` locally without Docker containers.

## 🎯 Quick Start

### Basic Local Execution
```bash
./ridges.py test-agent --agent-file miner/top_agent.py --num-problems 1 --problem-set easy --verbose --local
```

### With Debug Mode (Interactive Breakpoints)
```bash
./ridges.py test-agent --agent-file miner/top_agent.py --num-problems 1 --problem-set easy --verbose --debug --local
```

## 🐛 Debug Features

### 1. **Enhanced Logging**
- Debug messages appear in the terminal
- Detailed information about workflow transitions
- Parameter inspection and data flow tracking

### 2. **Interactive Breakpoints**
- Strategic breakpoints in key functions:
  - `multi_task_process` - Main entry point
  - `execute_test_patch_find_workflow_v1` - Test discovery phase
  - `execute_fix_workflow_v1` - Fix execution phase
- Use `debug_breakpoint("message")` to add custom breakpoints

### 3. **VS Code Debugging**
- Set breakpoints directly in VS Code
- Step through code with full IDE experience
- Inspect variables in real-time

## 🔧 How It Works

### Local Execution Flow:
1. **Repository Cloning**: Clones the target repository (e.g., psf/requests)
2. **Agent Setup**: Copies your agent code to a temporary directory
3. **Direct Execution**: Runs your agent code directly in the current Python environment
4. **No Docker**: Bypasses all Docker containers for faster execution

### Debug Flow:
1. **Environment Setup**: Sets `RIDGES_DEBUG=true` environment variable
2. **Debug Logging**: Shows detailed execution information
3. **Interactive Breakpoints**: Pauses execution for inspection
4. **Full IDE Support**: Works with VS Code debugger

## 📊 Example Output

```
🐛 Debug mode enabled - detailed logging will be shown in agent execution
🚀 Running agent locally without Docker (faster debugging)
Loading problem set...
Selected 1 problems from easy set
Running locally (no Docker containers)

Starting test 1/1: psf__requests-5414
   Running locally (no Docker)...
   Cloning repository: psf/requests...
   Repository ready at /tmp/ridges_local_agent_xxx/repo
   Agent code ready at /tmp/ridges_local_agent_xxx/agent/agent.py
   Running agent in directory: /tmp/ridges_local_agent_xxx/repo
   Executing agent...
🐛 DEBUG: === STARTING AGENT EXECUTION ===
🐛 DEBUG: Input dict keys: ['problem_statement', 'instance_id', 'repo_dir', 'run_id']
🐛 DEBUG: Repo dir: /tmp/ridges_local_agent_xxx/repo
🐛 DEBUG: Current working directory: /tmp/ridges_local_agent_xxx/repo
🐛 DEBUG: Problem text length: 1428
🐛 DEBUG: Instance ID: psf__requests-5414
🐛 BREAKPOINT: Starting multi_task_process - inspect input data
--Return--
> /tmp/ridges_local_agent_xxx/agent/agent.py(39)debug_breakpoint()->None
-> pdb.set_trace()
(Pdb) 
```

## 🎛️ Debug Commands

When you hit a breakpoint, you can use these commands:

```
(Pdb) l          # List current code
(Pdb) p variable_name  # Print variable
(Pdb) pp variable_name  # Pretty print variable
(Pdb) n          # Next line
(Pdb) s          # Step into function
(Pdb) c          # Continue execution
(Pdb) q          # Quit debugger
(Pdb) h          # Show help
```

## 🔍 Custom Debug Breakpoints

Add custom breakpoints anywhere in your code:

```python
# Add this to any function where you want to pause
debug_breakpoint("Custom breakpoint message")

# Or use debug_print for logging without stopping
debug_print("Variable value:", variable_name)
```

## 🚀 Benefits of Local Execution

1. **Faster Startup**: No Docker container creation
2. **Easier Debugging**: Direct access to your Python environment
3. **VS Code Integration**: Full IDE debugging support
4. **No Network Issues**: No proxy or container networking
5. **Immediate Feedback**: See results instantly
6. **Resource Efficient**: Uses your local Python environment

## 🎯 Best Practices

1. **Use `--local` flag** for development and debugging
2. **Use `--debug` flag** for interactive debugging
3. **Set breakpoints strategically** at key decision points
4. **Inspect variables** to understand data flow
5. **Use `c` (continue) to skip through known-good code**
6. **Use `n` (next) to step through line by line**

## 🔧 Troubleshooting

### Agent Code Not Found:
- Make sure the `--agent-file` path is correct
- Check that the file exists and is readable

### Repository Cloning Issues:
- Ensure you have `git` installed
- Check your internet connection
- Verify the repository URL is accessible

### Debug Mode Not Working:
- Make sure you're using the `--debug` flag
- Check that `RIDGES_DEBUG=true` is set in the environment

### VS Code Debugging:
- Set breakpoints in your agent code
- Use the Python debugger configuration
- Make sure the file paths match

Happy debugging! 🎉
