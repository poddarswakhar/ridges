# 🐍 PyCharm Debugger Guide for Ridges Agent

This guide shows you how to use PyCharm's powerful debugger with your Ridges agent for the best debugging experience.

## 🎯 **Method 1: Direct PyCharm Debugging (Recommended)**

### **Step 1: Set Up Run Configuration**

1. **Open PyCharm** and load your Ridges project
2. **Go to Run Configuration**:
   - Click `Run` → `Edit Configurations...`
   - Click `+` → `Python`
   - Fill in the configuration:

```
Name: Debug Ridges Agent
Script path: /Users/swakharpoddar/dev/ridges/ridges.py
Parameters: test-agent --agent-file miner/top_agent.py --num-problems 1 --problem-set easy --verbose --local
Working directory: /Users/swakharpoddar/dev/ridges
Environment variables: RIDGES_DEBUG=true
```

3. **Click OK** to save the configuration

### **Step 2: Set Breakpoints**

1. **Open `miner/top_agent.py`** in PyCharm
2. **Set breakpoints** by clicking in the left margin next to line numbers
3. **Recommended breakpoint locations**:
   - Line 8048: `debug_breakpoint("Starting multi_task_process - inspect input data")`
   - Line 8050: `if not problem_text:`
   - Any function you want to debug

### **Step 3: Start Debugging**

1. **Select your configuration** from the dropdown
2. **Click the debug button** (🐛) or press `Shift+F9`
3. **PyCharm will start debugging** and pause at your breakpoints

## 🚀 **Method 2: Remote Debugging with debugpy**

### **Step 1: Install debugpy**

```bash
pip install debugpy
```

### **Step 2: Set Up Remote Debugging**

1. **Create a new Run Configuration**:
   - `Run` → `Edit Configurations...`
   - `+` → `Python`
   - Fill in:

```
Name: Debug Ridges Agent (Remote)
Script path: /Users/swakharpoddar/dev/ridges/ridges.py
Parameters: test-agent --agent-file miner/top_agent.py --num-problems 1 --problem-set easy --verbose --debug --local
Working directory: /Users/swakharpoddar/dev/ridges
Environment variables: RIDGES_DEBUG=true;RIDGES_DEBUG_WAIT=true
```

2. **Create Remote Debug Configuration**:
   - `Run` → `Edit Configurations...`
   - `+` → `Python Debug Server`
   - Fill in:

```
Name: Ridges Remote Debug
Host: localhost
Port: 5678
Path mappings: 
  - Local path: /Users/swakharpoddar/dev/ridges/miner
  - Remote path: /sandbox/src
```

### **Step 3: Start Remote Debugging**

1. **Start the Remote Debug Server** (click debug button on "Ridges Remote Debug")
2. **Run your agent** with the remote debug configuration
3. **PyCharm will attach** to the running process

## 🎛️ **PyCharm Debugger Features**

### **Debug Controls**
- **F8** (Step Over): Execute current line, don't go into functions
- **F7** (Step Into): Go into function calls
- **Shift+F8** (Step Out): Exit current function
- **F9** (Resume): Continue execution until next breakpoint
- **Ctrl+F2** (Stop): Stop debugging

### **Variable Inspection**
- **Variables panel**: See all local variables
- **Watches**: Add expressions to monitor
- **Evaluate Expression**: Press `Alt+F8` to evaluate any expression

### **Call Stack**
- **Frames panel**: See the call stack
- **Click on frames** to jump to different execution points

### **Console**
- **Debug Console**: Execute Python code in the current context
- **Variables Console**: See variable values

## 🔍 **Advanced Debugging Techniques**

### **1. Conditional Breakpoints**
- **Right-click on breakpoint** → `Edit Breakpoint`
- **Add condition**: `len(problem_text) > 1000`
- **Breakpoint will only trigger** when condition is true

### **2. Log Points**
- **Right-click on breakpoint** → `Edit Breakpoint`
- **Check "Log message to console"**
- **Add message**: `Problem text length: {len(problem_text)}`
- **No pause, just logging**

### **3. Exception Breakpoints**
- **Run** → `View Breakpoints`
- **Click `+`** → `Python Exception Breakpoint`
- **Select exception types** to break on

### **4. Watches**
- **Right-click in Variables panel** → `Add Watch`
- **Add expressions**: `problem_text[:100]`, `len(input_dict)`

## 📊 **Example Debug Session**

### **What You'll See:**

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
🐛 DEBUG: Hints: None...
🐛 BREAKPOINT: Starting multi_task_process - inspect input data
```

### **PyCharm Debugger Panel:**
- **Variables**: `input_dict`, `problem_text`, `instance_id`
- **Watches**: `len(problem_text)`, `input_dict.keys()`
- **Call Stack**: `multi_task_process` → `_execute_agent` → `run_agent`

## 🎯 **Best Practices**

### **1. Strategic Breakpoints**
- **Set breakpoints at key decision points**
- **Use conditional breakpoints** for specific scenarios
- **Remove breakpoints** when not needed

### **2. Variable Inspection**
- **Use watches** for frequently accessed variables
- **Inspect complex objects** in the Variables panel
- **Use Evaluate Expression** for quick calculations

### **3. Step-by-Step Debugging**
- **Use F8 (Step Over)** for most debugging
- **Use F7 (Step Into)** only when needed
- **Use F9 (Resume)** to skip known-good code

### **4. Exception Handling**
- **Set exception breakpoints** for unexpected errors
- **Inspect exception details** in the Variables panel
- **Use the call stack** to trace error origins

## 🔧 **Troubleshooting**

### **Breakpoints Not Hitting**
- **Check file paths** match exactly
- **Ensure code is being executed**
- **Verify breakpoints are in the right file**

### **Variables Not Showing**
- **Check variable scope**
- **Use Evaluate Expression** for complex objects
- **Add watches** for better visibility

### **Remote Debugging Issues**
- **Check port 5678** is available
- **Verify debugpy** is installed
- **Check environment variables** are set

## 🎉 **Benefits of PyCharm Debugging**

1. **Visual Interface**: Easy to see variables and call stack
2. **Advanced Features**: Conditional breakpoints, watches, etc.
3. **Integration**: Works seamlessly with your IDE
4. **Performance**: Fast debugging without Docker overhead
5. **Flexibility**: Multiple debugging methods available

Happy debugging with PyCharm! 🐍✨
