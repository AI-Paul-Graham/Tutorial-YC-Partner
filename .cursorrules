================================================
File: docs/guide.md
================================================
---
layout: default
title: "Build your LLM App"
---

# LLM Application Development Playbook

> If you are an AI assistant involved in building LLM Apps, read this guide **VERY, VERY** carefully! This is the most important chapter in the entire document. Throughout development, you should always (1) start with a small and simple solution, (2) design at a high level (`docs/design.md`) before implementation, and (3) frequently ask humans for feedback and clarification.
{: .warning }

## System Design Steps

These system designs should be a collaboration between humans and AI assistants:

| Stage                  | Human      | AI        | Comment                                                                 |
|:-----------------------|:----------:|:---------:|:------------------------------------------------------------------------|
| 1. Project Requirements | ★★★ High  | ★☆☆ Low   | Humans understand the requirements and context best.                    |
| 2. Utility Functions   | ★★☆ Medium | ★★☆ Medium | The human is familiar with external APIs and integrations, and the AI assists with implementation. |
| 3. Flow Design         | ★★☆ Medium | ★★☆ Medium | The human identifies complex and ambiguous parts, and the AI helps with redesign. |
| 4. Data Schema         | ★☆☆ Low   | ★★★ High  | The AI assists in designing the data schema based on the flow.          |
| 5. Implementation      | ★☆☆ Low   | ★★★ High  | The human identifies complex and ambiguous parts, and the AI helps with redesign. |
| 6. Optimization        | ★★☆ Medium | ★★☆ Medium | The human reviews the code and evaluates the results, while the AI helps optimize. |
| 7. Reliability         | ★☆☆ Low   | ★★★ High  | The AI helps write test cases and address corner cases.                 |

1. **Project Requirements**: Clarify the requirements for your project, and evaluate whether an AI system is a good fit. An AI systems are:
    - suitable for routine tasks that require common sense (e.g., filling out forms, replying to emails).
    - suitable for creative tasks where all inputs are provided (e.g., building slides, writing SQL).
    - **NOT** suitable for tasks that are highly ambiguous and require complex information (e.g., building a startup).
    - > **If a human can’t solve it, an LLM can’t automate it!** Before building an LLM system, thoroughly understand the problem by manually solving example inputs to develop intuition.  
      {: .best-practice }

2. **Utility Functions**: AI system is the decision-maker and relies on *external utility functions* to:

   <div align="center"><img src="https://github.com/the-pocket/PocketFlow/raw/main/assets/utility.png?raw=true" width="400"/></div>

   - Read inputs (e.g., retrieving Slack messages, reading emails)
   - Write outputs (e.g., generating reports, sending emails)
   - Use external tools (e.g., calling LLMs, searching the web)
   - In contrast, *LLM-based tasks* (e.g., summarizing text, analyzing sentiment) are **NOT** utility functions. Instead, they are *internal core functions* within the AI system—designed in step 3—and are built on top of the utility functions.
   -  > **Start small!** Only include the most important ones to begin with!
      {: .best-practice }

3. **Flow Design (Compute)**: Create a high-level outline for your application’s flow.
    - Identify potential design patterns (e.g., Batch, Agent, RAG).
    - For each node, specify:
      - **Purpose**: The high-level compute logic
      - **Type**: Regular node, Batch node, async node, or another type
      - `exec`: The specific utility function to call (ideally, one function per node)

4. **Data Schema (Data)**: Plan how data will be stored and updated.
   - For simple apps, use an in-memory dictionary.
   - For more complex apps or when persistence is required, use a database.
   - For each node, specify:
     - `prep`: How the node reads data
     - `post`: How the node writes data

5. **Implementation**: Implement nodes and flows based on the design.
   - Start with a simple, direct approach (avoid over-engineering and full-scale type checking or testing). Let it fail fast to identify weaknesses.
   - Add logging throughout the code to facilitate debugging.

6. **Optimization**:
   - **Use Intuition**: For a quick initial evaluation, human intuition is often a good start.
   - **Redesign Flow (Back to Step 3)**: Consider breaking down tasks further, introducing agentic decisions, or better managing input contexts.
   - If your flow design is already solid, move on to micro-optimizations:
     - **Prompt Engineering**: Use clear, specific instructions with examples to reduce ambiguity.
     - **In-Context Learning**: Provide robust examples for tasks that are difficult to specify with instructions alone.

   - > **You’ll likely iterate a lot!** Expect to repeat Steps 3–6 hundreds of times.
     >
     > <div align="center"><img src="https://github.com/the-pocket/PocketFlow/raw/main/assets/success.png?raw=true" width="400"/></div>
     {: .best-practice }

7. **Reliability**  
   - **Node Retries**: Add checks in the node `exec` to ensure outputs meet requirements, and consider increasing `max_retries` and `wait` times.
   - **Logging and Visualization**: Maintain logs of all attempts and visualize node results for easier debugging.
   - **Self-Evaluation**: Add a separate node (powered by an LLM) to review outputs when results are uncertain.

## Example LLM Project File Structure

```
my_project/
├── main.py
├── flow.py
├── utils/
│   ├── __init__.py
│   ├── call_llm.py
│   └── search_web.py
├── requirements.txt
└── docs/
    └── design.md
```

- **`docs/design.md`**: Contains project documentation for each step above. This should be high-level and no-code.
- **`utils/`**: Contains all utility functions.
  - It’s recommended to dedicate one Python file to each API call, for example `call_llm.py` or `search_web.py`.
  - Each file should also include a `main()` function to try that API call
- **`flow.py`**: Implements the application’s flow, starting with node definitions followed by the overall structure.
- **`main.py`**: Serves as the project’s entry point.

================================================
File: docs/index.md
================================================
---
layout: default
title: "Home"
nav_order: 1
---

# Pocket Flow

A [100-line](https://github.com/the-pocket/PocketFlow/blob/main/pocketflow/__init__.py) minimalist LLM framework for *Agents, Task Decomposition, RAG, etc*.

We model the LLM workflow as a **Graph + Shared Store**:
- **Nodes** handle simple (LLM) tasks.
- Nodes connect through **Actions** (labeled edges) for *Agents*.  
- **Flows** orchestrate a directed graph of Nodes for *Task Decomposition*.
- Nodes within Flows communicate through a **Shared Store**.
- A Flow can be used as a Node (for **Nesting**).
- **Batch** Nodes/Flows for data-intensive tasks.
- **Async** Nodes/Flows allow waits for asynchronous tasks.

<div align="center">
  <img src="https://github.com/the-pocket/PocketFlow/raw/main/assets/meme.jpg?raw=true" width="400"/>
</div>

> If you are an AI assistant reading this document, please read all the non-optional and non-advanced parts. If you are helping a human build an LLM app, please carefully review the [Development Playbook](./guide.md).
{: .warning }

## Core Abstraction

- [Node](./core_abstraction/node.md)
- [Flow](./core_abstraction/flow.md)
- [Communication](./core_abstraction/communication.md)
- [Batch](./core_abstraction/batch.md)
- [(Advanced) Async](./core_abstraction/async.md)
- [(Advanced) Parallel](./core_abstraction/parallel.md)

## Utility Function

- [LLM Wrapper](./utility_function/llm.md)
- [Tool](./utility_function/tool.md)
- [(Optional) Viz and Debug](./utility_function/viz.md)
- Chunking

> We do not provide built-in utility functions. Example implementations are provided as reference.
{: .warning }


## Design Pattern

- [Structured Output](./design_pattern/structure.md)
- [Workflow](./design_pattern/workflow.md)
- [Map Reduce](./design_pattern/mapreduce.md)
- [RAG](./design_pattern/rag.md)
- [Agent](./design_pattern/agent.md)
- [(Optional) Chat Memory](./design_pattern/memory.md)
- [(Advanced) Multi-Agents](./design_pattern/multi_agent.md)
- Evaluation

## [Develop your LLM Apps](./guide.md)

================================================
File: docs/core_abstraction/async.md
================================================
---
layout: default
title: "(Advanced) Async"
parent: "Core Abstraction"
nav_order: 5
---

# (Advanced) Async

**Async** Nodes implement `prep_async()`, `exec_async()`, `exec_fallback_async()`, and/or `post_async()`. This is useful for:

1. **prep_async()**: For *fetching/reading data (files, APIs, DB)* in an I/O-friendly way.
2. **exec_async()**: Typically used for async LLM calls.
3. **post_async()**: For *awaiting user feedback*, *coordinating across multi-agents* or any additional async steps after `exec_async()`.

**Note**: `AsyncNode` must be wrapped in `AsyncFlow`. `AsyncFlow` can also include regular (sync) nodes.

### Example

```python
class SummarizeThenVerify(AsyncNode):
    async def prep_async(self, shared):
        # Example: read a file asynchronously
        doc_text = await read_file_async(shared["doc_path"])
        return doc_text

    async def exec_async(self, prep_res):
        # Example: async LLM call
        summary = await call_llm_async(f"Summarize: {prep_res}")
        return summary

    async def post_async(self, shared, prep_res, exec_res):
        # Example: wait for user feedback
        decision = await gather_user_feedback(exec_res)
        if decision == "approve":
            shared["summary"] = exec_res
            return "approve"
        return "deny"

summarize_node = SummarizeThenVerify()
final_node = Finalize()

# Define transitions
summarize_node - "approve" >> final_node
summarize_node - "deny"    >> summarize_node  # retry

flow = AsyncFlow(start=summarize_node)

async def main():
    shared = {"doc_path": "document.txt"}
    await flow.run_async(shared)
    print("Final Summary:", shared.get("summary"))

asyncio.run(main())
```

================================================
File: docs/core_abstraction/batch.md
================================================
---
layout: default
title: "Batch"
parent: "Core Abstraction"
nav_order: 4
---

# Batch

**Batch** makes it easier to handle large inputs in one Node or **rerun** a Flow multiple times. Example use cases:
- **Chunk-based** processing (e.g., splitting large texts).
- **Iterative** processing over lists of input items (e.g., user queries, files, URLs).

## 1. BatchNode

A **BatchNode** extends `Node` but changes `prep()` and `exec()`:

- **`prep(shared)`**: returns an **iterable** (e.g., list, generator).
- **`exec(item)`**: called **once** per item in that iterable.
- **`post(shared, prep_res, exec_res_list)`**: after all items are processed, receives a **list** of results (`exec_res_list`) and returns an **Action**.


### Example: Summarize a Large File

```python
class MapSummaries(BatchNode):
    def prep(self, shared):
        # Suppose we have a big file; chunk it
        content = shared["data"]
        chunk_size = 10000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        return chunks

    def exec(self, chunk):
        prompt = f"Summarize this chunk in 10 words: {chunk}"
        summary = call_llm(prompt)
        return summary

    def post(self, shared, prep_res, exec_res_list):
        combined = "\n".join(exec_res_list)
        shared["summary"] = combined
        return "default"

map_summaries = MapSummaries()
flow = Flow(start=map_summaries)
flow.run(shared)
```

---

## 2. BatchFlow

A **BatchFlow** runs a **Flow** multiple times, each time with different `params`. Think of it as a loop that replays the Flow for each parameter set.


### Example: Summarize Many Files

```python
class SummarizeAllFiles(BatchFlow):
    def prep(self, shared):
        # Return a list of param dicts (one per file)
        filenames = list(shared["data"].keys())  # e.g., ["file1.txt", "file2.txt", ...]
        return [{"filename": fn} for fn in filenames]

# Suppose we have a per-file Flow (e.g., load_file >> summarize >> reduce):
summarize_file = SummarizeFile(start=load_file)

# Wrap that flow into a BatchFlow:
summarize_all_files = SummarizeAllFiles(start=summarize_file)
summarize_all_files.run(shared)
```

### Under the Hood
1. `prep(shared)` returns a list of param dicts—e.g., `[{filename: "file1.txt"}, {filename: "file2.txt"}, ...]`.
2. The **BatchFlow** loops through each dict. For each one:
   - It merges the dict with the BatchFlow’s own `params`.
   - It calls `flow.run(shared)` using the merged result.
3. This means the sub-Flow is run **repeatedly**, once for every param dict.

---

## 3. Nested or Multi-Level Batches

You can nest a **BatchFlow** in another **BatchFlow**. For instance:
- **Outer** batch: returns a list of diretory param dicts (e.g., `{"directory": "/pathA"}`, `{"directory": "/pathB"}`, ...).
- **Inner** batch: returning a list of per-file param dicts.

At each level, **BatchFlow** merges its own param dict with the parent’s. By the time you reach the **innermost** node, the final `params` is the merged result of **all** parents in the chain. This way, a nested structure can keep track of the entire context (e.g., directory + file name) at once.

```python

class FileBatchFlow(BatchFlow):
    def prep(self, shared):
        directory = self.params["directory"]
        # e.g., files = ["file1.txt", "file2.txt", ...]
        files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        return [{"filename": f} for f in files]

class DirectoryBatchFlow(BatchFlow):
    def prep(self, shared):
        directories = [ "/path/to/dirA", "/path/to/dirB"]
        return [{"directory": d} for d in directories]

# MapSummaries have params like {"directory": "/path/to/dirA", "filename": "file1.txt"}
inner_flow = FileBatchFlow(start=MapSummaries())
outer_flow = DirectoryBatchFlow(start=inner_flow)
```

================================================
File: docs/core_abstraction/communication.md
================================================
---
layout: default
title: "Communication"
parent: "Core Abstraction"
nav_order: 3
---

# Communication

Nodes and Flows **communicate** in two ways:

1. **Shared Store (recommended)** 

   - A global data structure (often an in-mem dict) that all nodes can read and write by `prep()` and `post()`.  
   - Great for data results, large content, or anything multiple nodes need.
   - You shall design the data structure and populate it ahead.

2. **Params (only for [Batch](./batch.md))** 
   - Each node has a local, ephemeral `params` dict passed in by the **parent Flow**, used as an identifier for tasks. Parameter keys and values shall be **immutable**.
   - Good for identifiers like filenames or numeric IDs, in Batch mode.

If you know memory management, think of the **Shared Store** like a **heap** (shared by all function calls), and **Params** like a **stack** (assigned by the caller).

> Use `Shared Store` for almost all cases. It's flexible and easy to manage. It separates *Data Schema* from *Compute Logic*, making the code easier to maintain. `Params` is more a syntax sugar for [Batch](./batch.md).
{: .best-practice }

---

## 1. Shared Store

### Overview

A shared store is typically an in-mem dictionary, like:
```python
shared = {"data": {}, "summary": {}, "config": {...}, ...}
```

It can also contain local file handlers, DB connections, or a combination for persistence. We recommend deciding the data structure or DB schema first based on your app requirements.

### Example

```python
class LoadData(Node):
    def post(self, shared, prep_res, exec_res):
        # We write data to shared store
        shared["data"] = "Some text content"
        return None

class Summarize(Node):
    def prep(self, shared):
        # We read data from shared store
        return shared["data"]

    def exec(self, prep_res):
        # Call LLM to summarize
        prompt = f"Summarize: {prep_res}"
        summary = call_llm(prompt)
        return summary

    def post(self, shared, prep_res, exec_res):
        # We write summary to shared store
        shared["summary"] = exec_res
        return "default"

load_data = LoadData()
summarize = Summarize()
load_data >> summarize
flow = Flow(start=load_data)

shared = {}
flow.run(shared)
```

Here:
- `LoadData` writes to `shared["data"]`.
- `Summarize` reads from `shared["data"]`, summarizes, and writes to `shared["summary"]`.

---

## 2. Params

**Params** let you store *per-Node* or *per-Flow* config that doesn't need to live in the shared store. They are:
- **Immutable** during a Node's run cycle (i.e., they don't change mid-`prep->exec->post`).
- **Set** via `set_params()`.
- **Cleared** and updated each time a parent Flow calls it.

> Only set the uppermost Flow params because others will be overwritten by the parent Flow. 
> 
> If you need to set child node params, see [Batch](./batch.md).
{: .warning }

Typically, **Params** are identifiers (e.g., file name, page number). Use them to fetch the task you assigned or write to a specific part of the shared store.

### Example

```python
# 1) Create a Node that uses params
class SummarizeFile(Node):
    def prep(self, shared):
        # Access the node's param
        filename = self.params["filename"]
        return shared["data"].get(filename, "")

    def exec(self, prep_res):
        prompt = f"Summarize: {prep_res}"
        return call_llm(prompt)

    def post(self, shared, prep_res, exec_res):
        filename = self.params["filename"]
        shared["summary"][filename] = exec_res
        return "default"

# 2) Set params
node = SummarizeFile()

# 3) Set Node params directly (for testing)
node.set_params({"filename": "doc1.txt"})
node.run(shared)

# 4) Create Flow
flow = Flow(start=node)

# 5) Set Flow params (overwrites node params)
flow.set_params({"filename": "doc2.txt"})
flow.run(shared)  # The node summarizes doc2, not doc1
```

================================================
File: docs/core_abstraction/flow.md
================================================
---
layout: default
title: "Flow"
parent: "Core Abstraction"
nav_order: 2
---

# Flow

A **Flow** orchestrates a graph of Nodes. You can chain Nodes in a sequence or create branching depending on the **Actions** returned from each Node's `post()`.

## 1. Action-based Transitions

Each Node's `post()` returns an **Action** string. By default, if `post()` doesn't return anything, we treat that as `"default"`.

You define transitions with the syntax:

1. **Basic default transition**: `node_a >> node_b`
  This means if `node_a.post()` returns `"default"`, go to `node_b`. 
  (Equivalent to `node_a - "default" >> node_b`)

2. **Named action transition**: `node_a - "action_name" >> node_b`
  This means if `node_a.post()` returns `"action_name"`, go to `node_b`.

It's possible to create loops, branching, or multi-step flows.

## 2. Creating a Flow

A **Flow** begins with a **start** node. You call `Flow(start=some_node)` to specify the entry point. When you call `flow.run(shared)`, it executes the start node, looks at its returned Action from `post()`, follows the transition, and continues until there's no next node.

### Example: Simple Sequence

Here's a minimal flow of two nodes in a chain:

```python
node_a >> node_b
flow = Flow(start=node_a)
flow.run(shared)
```

- When you run the flow, it executes `node_a`.  
- Suppose `node_a.post()` returns `"default"`.  
- The flow then sees `"default"` Action is linked to `node_b` and runs `node_b`.  
- `node_b.post()` returns `"default"` but we didn't define `node_b >> something_else`. So the flow ends there.

### Example: Branching & Looping

Here's a simple expense approval flow that demonstrates branching and looping. The `ReviewExpense` node can return three possible Actions:

- `"approved"`: expense is approved, move to payment processing
- `"needs_revision"`: expense needs changes, send back for revision 
- `"rejected"`: expense is denied, finish the process

We can wire them like this:

```python
# Define the flow connections
review - "approved" >> payment        # If approved, process payment
review - "needs_revision" >> revise   # If needs changes, go to revision
review - "rejected" >> finish         # If rejected, finish the process

revise >> review   # After revision, go back for another review
payment >> finish  # After payment, finish the process

flow = Flow(start=review)
```

Let's see how it flows:

1. If `review.post()` returns `"approved"`, the expense moves to the `payment` node
2. If `review.post()` returns `"needs_revision"`, it goes to the `revise` node, which then loops back to `review`
3. If `review.post()` returns `"rejected"`, it moves to the `finish` node and stops

```mermaid
flowchart TD
    review[Review Expense] -->|approved| payment[Process Payment]
    review -->|needs_revision| revise[Revise Report]
    review -->|rejected| finish[Finish Process]

    revise --> review
    payment --> finish
```

### Running Individual Nodes vs. Running a Flow

- `node.run(shared)`: Just runs that node alone (calls `prep->exec->post()`), returns an Action. 
- `flow.run(shared)`: Executes from the start node, follows Actions to the next node, and so on until the flow can't continue.

> `node.run(shared)` **does not** proceed to the successor.
> This is mainly for debugging or testing a single node.
> 
> Always use `flow.run(...)` in production to ensure the full pipeline runs correctly.
{: .warning }

## 3. Nested Flows

A **Flow** can act like a Node, which enables powerful composition patterns. This means you can:

1. Use a Flow as a Node within another Flow's transitions.  
2. Combine multiple smaller Flows into a larger Flow for reuse.  
3. Node `params` will be a merging of **all** parents' `params`.

### Flow's Node Methods

A **Flow** is also a **Node**, so it will run `prep()` and `post()`. However:

- It **won't** run `exec()`, as its main logic is to orchestrate its nodes.
- `post()` always receives `None` for `exec_res` and should instead get the flow execution results from the shared store.

### Basic Flow Nesting

Here's how to connect a flow to another node:

```python
# Create a sub-flow
node_a >> node_b
subflow = Flow(start=node_a)

# Connect it to another node
subflow >> node_c

# Create the parent flow
parent_flow = Flow(start=subflow)
```

When `parent_flow.run()` executes:
1. It starts `subflow`
2. `subflow` runs through its nodes (`node_a->node_b`)
3. After `subflow` completes, execution continues to `node_c`

### Example: Order Processing Pipeline

Here's a practical example that breaks down order processing into nested flows:

```python
# Payment processing sub-flow
validate_payment >> process_payment >> payment_confirmation
payment_flow = Flow(start=validate_payment)

# Inventory sub-flow
check_stock >> reserve_items >> update_inventory
inventory_flow = Flow(start=check_stock)

# Shipping sub-flow
create_label >> assign_carrier >> schedule_pickup
shipping_flow = Flow(start=create_label)

# Connect the flows into a main order pipeline
payment_flow >> inventory_flow >> shipping_flow

# Create the master flow
order_pipeline = Flow(start=payment_flow)

# Run the entire pipeline
order_pipeline.run(shared_data)
```

This creates a clean separation of concerns while maintaining a clear execution path:

```mermaid
flowchart LR
    subgraph order_pipeline[Order Pipeline]
        subgraph paymentFlow["Payment Flow"]
            A[Validate Payment] --> B[Process Payment] --> C[Payment Confirmation]
        end

        subgraph inventoryFlow["Inventory Flow"]
            D[Check Stock] --> E[Reserve Items] --> F[Update Inventory]
        end

        subgraph shippingFlow["Shipping Flow"]
            G[Create Label] --> H[Assign Carrier] --> I[Schedule Pickup]
        end

        paymentFlow --> inventoryFlow
        inventoryFlow --> shippingFlow
    end
```

================================================
File: docs/core_abstraction/node.md
================================================
---
layout: default
title: "Node"
parent: "Core Abstraction"
nav_order: 1
---

# Node

A **Node** is the smallest building block. Each Node has 3 steps `prep->exec->post`:

<div align="center">
  <img src="https://github.com/the-pocket/PocketFlow/raw/main/assets/node.png?raw=true" width="400"/>
</div>

1. `prep(shared)`
   - **Read and preprocess data** from `shared` store. 
   - Examples: *query DB, read files, or serialize data into a string*.
   - Return `prep_res`, which is used by `exec()` and `post()`.

2. `exec(prep_res)`
   - **Execute compute logic**, with optional retries and error handling (below).
   - Examples: *(mostly) LLM calls, remote APIs, tool use*.
   - ⚠️ This shall be only for compute and **NOT** access `shared`.
   - ⚠️ If retries enabled, ensure idempotent implementation.
   - Return `exec_res`, which is passed to `post()`.

3. `post(shared, prep_res, exec_res)`
   - **Postprocess and write data** back to `shared`.
   - Examples: *update DB, change states, log results*.
   - **Decide the next action** by returning a *string* (`action = "default"` if *None*).

> **Why 3 steps?** To enforce the principle of *separation of concerns*. The data storage and data processing are operated separately.
>
> All steps are *optional*. E.g., you can only implement `prep` and `post` if you just need to process data.
{: .note }

### Fault Tolerance & Retries

You can **retry** `exec()` if it raises an exception via two parameters when define the Node:

- `max_retries` (int): Max times to run `exec()`. The default is `1` (**no** retry).
- `wait` (int): The time to wait (in **seconds**) before next retry. By default, `wait=0` (no waiting). 
`wait` is helpful when you encounter rate-limits or quota errors from your LLM provider and need to back off.

```python 
my_node = SummarizeFile(max_retries=3, wait=10)
```

When an exception occurs in `exec()`, the Node automatically retries until:

- It either succeeds, or
- The Node has retried `max_retries - 1` times already and fails on the last attempt.

You can get the current retry times (0-based) from `self.cur_retry`.

```python 
class RetryNode(Node):
    def exec(self, prep_res):
        print(f"Retry {self.cur_retry} times")
        raise Exception("Failed")
```

### Graceful Fallback

To **gracefully handle** the exception (after all retries) rather than raising it, override:

```python 
def exec_fallback(self, prep_res, exc):
    raise exc
```

By default, it just re-raises exception. But you can return a fallback result instead, which becomes the `exec_res` passed to `post()`.

### Example: Summarize file

```python 
class SummarizeFile(Node):
    def prep(self, shared):
        return shared["data"]

    def exec(self, prep_res):
        if not prep_res:
            return "Empty file content"
        prompt = f"Summarize this text in 10 words: {prep_res}"
        summary = call_llm(prompt)  # might fail
        return summary

    def exec_fallback(self, prep_res, exc):
        # Provide a simple fallback instead of crashing
        return "There was an error processing your request."

    def post(self, shared, prep_res, exec_res):
        shared["summary"] = exec_res
        # Return "default" by not returning

summarize_node = SummarizeFile(max_retries=3)

# node.run() calls prep->exec->post
# If exec() fails, it retries up to 3 times before calling exec_fallback()
action_result = summarize_node.run(shared)

print("Action returned:", action_result)  # "default"
print("Summary stored:", shared["summary"])
```

================================================
File: docs/core_abstraction/parallel.md
================================================
---
layout: default
title: "(Advanced) Parallel"
parent: "Core Abstraction"
nav_order: 6
---

# (Advanced) Parallel

**Parallel** Nodes and Flows let you run multiple **Async** Nodes and Flows  **concurrently**—for example, summarizing multiple texts at once. This can improve performance by overlapping I/O and compute. 

> Because of Python’s GIL, parallel nodes and flows can’t truly parallelize CPU-bound tasks (e.g., heavy numerical computations). However, they excel at overlapping I/O-bound work—like LLM calls, database queries, API requests, or file I/O.
{: .warning }

> - **Ensure Tasks Are Independent**: If each item depends on the output of a previous item, **do not** parallelize.
> 
> - **Beware of Rate Limits**: Parallel calls can **quickly** trigger rate limits on LLM services. You may need a **throttling** mechanism (e.g., semaphores or sleep intervals).
> 
> - **Consider Single-Node Batch APIs**: Some LLMs offer a **batch inference** API where you can send multiple prompts in a single call. This is more complex to implement but can be more efficient than launching many parallel requests and mitigates rate limits.
{: .best-practice }

## AsyncParallelBatchNode

Like **AsyncBatchNode**, but run `exec_async()` in **parallel**:

```python
class ParallelSummaries(AsyncParallelBatchNode):
    async def prep_async(self, shared):
        # e.g., multiple texts
        return shared["texts"]

    async def exec_async(self, text):
        prompt = f"Summarize: {text}"
        return await call_llm_async(prompt)

    async def post_async(self, shared, prep_res, exec_res_list):
        shared["summary"] = "\n\n".join(exec_res_list)
        return "default"

node = ParallelSummaries()
flow = AsyncFlow(start=node)
```

## AsyncParallelBatchFlow

Parallel version of **BatchFlow**. Each iteration of the sub-flow runs **concurrently** using different parameters:

```python
class SummarizeMultipleFiles(AsyncParallelBatchFlow):
    async def prep_async(self, shared):
        return [{"filename": f} for f in shared["files"]]

sub_flow = AsyncFlow(start=LoadAndSummarizeFile())
parallel_flow = SummarizeMultipleFiles(start=sub_flow)
await parallel_flow.run_async(shared)
```

================================================
File: docs/design_pattern/agent.md
================================================
---
layout: default
title: "Agent"
parent: "Design Pattern"
nav_order: 6
---

# Agent

Agent is a powerful design pattern, where node can take dynamic actions based on the context it receives.
To express an agent, create a Node (the agent) with [branching](../core_abstraction/flow.md) to other nodes (Actions).

> The core of build **performant** and **reliable** agents boils down to:
> 
> 1. **Context Management:** Provide *clear, relevant context* so agents can understand the problem.E.g., Rather than dumping an entire chat history or entire files, use a [Workflow](./workflow.md) that filters out and includes only the most relevant information.
>
> 2. **Action Space:** Define *a well-structured, unambiguous, and easy-to-use* set of actions. For instance, avoid creating overlapping actions like `read_databases` and `read_csvs`. Instead, unify data sources (e.g., move CSVs into a database) and design a single action. The action can be parameterized (e.g., string for search) or  programmable (e.g., SQL queries).
{: .best-practice }

### Example: Search Agent

This agent:
1. Decides whether to search or answer
2. If searches, loops back to decide if more search needed
3. Answers when enough context gathered

```python
class DecideAction(Node):
    def prep(self, shared):
        context = shared.get("context", "No previous search")
        query = shared["query"]
        return query, context
        
    def exec(self, inputs):
        query, context = inputs
        prompt = f"""
Given input: {query}
Previous search results: {context}
Should I: 1) Search web for more info 2) Answer with current knowledge
Output in yaml:
```yaml
action: search/answer
reason: why this action
search_term: search phrase if action is search
```"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)
        
        assert isinstance(result, dict)
        assert "action" in result
        assert "reason" in result
        assert result["action"] in ["search", "answer"]
        if result["action"] == "search":
            assert "search_term" in result
        
        return result

    def post(self, shared, prep_res, exec_res):
        if exec_res["action"] == "search":
            shared["search_term"] = exec_res["search_term"]
        return exec_res["action"]

class SearchWeb(Node):
    def prep(self, shared):
        return shared["search_term"]
        
    def exec(self, search_term):
        return search_web(search_term)
    
    def post(self, shared, prep_res, exec_res):
        prev_searches = shared.get("context", [])
        shared["context"] = prev_searches + [
            {"term": shared["search_term"], "result": exec_res}
        ]
        return "decide"
        
class DirectAnswer(Node):
    def prep(self, shared):
        return shared["query"], shared.get("context", "")
        
    def exec(self, inputs):
        query, context = inputs
        return call_llm(f"Context: {context}\nAnswer: {query}")

    def post(self, shared, prep_res, exec_res):
       print(f"Answer: {exec_res}")
       shared["answer"] = exec_res

# Connect nodes
decide = DecideAction()
search = SearchWeb()
answer = DirectAnswer()

decide - "search" >> search
decide - "answer" >> answer
search - "decide" >> decide  # Loop back

flow = Flow(start=decide)
flow.run({"query": "Who won the Nobel Prize in Physics 2024?"})
```

================================================
File: docs/design_pattern/mapreduce.md
================================================
---
layout: default
title: "Map Reduce"
parent: "Design Pattern"
nav_order: 3
---

# Map Reduce

MapReduce is a design pattern suitable when you have either:
- Large input data (e.g., multiple files to process), or
- Large output data (e.g., multiple forms to fill)

and there is a logical way to break the task into smaller, ideally independent parts. 
You first break down the task using [BatchNode](../core_abstraction/batch.md) in the map phase, followed by aggregation in the reduce phase.

### Example: Document Summarization

```python
class MapSummaries(BatchNode):
    def prep(self, shared): return [shared["text"][i:i+10000] for i in range(0, len(shared["text"]), 10000)]
    def exec(self, chunk): return call_llm(f"Summarize this chunk: {chunk}")
    def post(self, shared, prep_res, exec_res_list): shared["summaries"] = exec_res_list

class ReduceSummaries(Node):
    def prep(self, shared): return shared["summaries"]
    def exec(self, summaries): return call_llm(f"Combine these summaries: {summaries}")
    def post(self, shared, prep_res, exec_res): shared["final_summary"] = exec_res

# Connect nodes
map_node = MapSummaries()
reduce_node = ReduceSummaries()
map_node >> reduce_node

# Create flow 
summarize_flow = Flow(start=map_node)
summarize_flow.run(shared)
```

================================================
File: docs/design_pattern/memory.md
================================================
---
layout: default
title: "Chat Memory"
parent: "Design Pattern"
nav_order: 5
---

# Chat Memory

Multi-turn conversations require memory management to maintain context while avoiding overwhelming the LLM.

### 1. Naive Approach: Full History

Sending the full chat history may overwhelm LLMs.

```python
class ChatNode(Node):
    def prep(self, shared):
        if "history" not in shared:
            shared["history"] = []
        user_input = input("You: ")
        return shared["history"], user_input

    def exec(self, inputs):
        history, user_input = inputs
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": user_input})
        response = call_llm(messages)
        return response

    def post(self, shared, prep_res, exec_res):
        shared["history"].append({"role": "user", "content": prep_res[1]})
        shared["history"].append({"role": "assistant", "content": exec_res})
        return "continue"

chat = ChatNode()
chat - "continue" >> chat
flow = Flow(start=chat)
```

### 2. Improved Memory Management

We can:
1. Limit the chat history to the most recent 4.
2. Use [vector search](./tool.md) to retrieve relevant exchanges beyond the last 4.

```python
################################
# Node A: Retrieve user input & relevant messages
################################
class ChatRetrieve(Node):
    def prep(self, s):
        s.setdefault("history", [])
        s.setdefault("memory_index", None)
        user_input = input("You: ")
        return user_input

    def exec(self, user_input):
        emb = get_embedding(user_input)
        relevant = []
        if len(shared["history"]) > 8 and shared["memory_index"]:
            idx, _ = search_index(shared["memory_index"], emb, top_k=2)
            relevant = [shared["history"][i[0]] for i in idx]
        return (user_input, relevant)

    def post(self, s, p, r):
        user_input, relevant = r
        s["user_input"] = user_input
        s["relevant"] = relevant
        return "continue"

################################
# Node B: Call LLM, update history + index
################################
class ChatReply(Node):
    def prep(self, s):
        user_input = s["user_input"]
        recent = s["history"][-8:]
        relevant = s.get("relevant", [])
        return user_input, recent, relevant

    def exec(self, inputs):
        user_input, recent, relevant = inputs
        msgs = [{"role":"system","content":"You are a helpful assistant."}]
        if relevant:
            msgs.append({"role":"system","content":f"Relevant: {relevant}"})
        msgs.extend(recent)
        msgs.append({"role":"user","content":user_input})
        ans = call_llm(msgs)
        return ans

    def post(self, s, pre, ans):
        user_input, _, _ = pre
        s["history"].append({"role":"user","content":user_input})
        s["history"].append({"role":"assistant","content":ans})
        
        # Manage memory index
        if len(s["history"]) == 8:
            embs = []
            for i in range(0, 8, 2):
                text = s["history"][i]["content"] + " " + s["history"][i+1]["content"]
                embs.append(get_embedding(text))
            s["memory_index"] = create_index(embs)
        elif len(s["history"]) > 8:
            text = s["history"][-2]["content"] + " " + s["history"][-1]["content"]
            new_emb = np.array([get_embedding(text)]).astype('float32')
            s["memory_index"].add(new_emb)

        print(f"Assistant: {ans}")
        return "continue"

################################
# Flow wiring
################################
retrieve = ChatRetrieve()
reply = ChatReply()
retrieve - "continue" >> reply
reply - "continue" >> retrieve

flow = Flow(start=retrieve)
shared = {}
flow.run(shared)
```

================================================
File: docs/design_pattern/multi_agent.md
================================================
---
layout: default
title: "(Advanced) Multi-Agents"
parent: "Design Pattern"
nav_order: 7
---

# (Advanced) Multi-Agents

Multiple [Agents](./flow.md) can work together by handling subtasks and communicating the progress. 
Communication between agents is typically implemented using message queues in shared storage.

> Most of time, you don't need Multi-Agents. Start with a simple solution first.
{: .best-practice }

### Example Agent Communication: Message Queue

Here's a simple example showing how to implement agent communication using `asyncio.Queue`. 
The agent listens for messages, processes them, and continues listening:

```python
class AgentNode(AsyncNode):
    async def prep_async(self, _):
        message_queue = self.params["messages"]
        message = await message_queue.get()
        print(f"Agent received: {message}")
        return message

# Create node and flow
agent = AgentNode()
agent >> agent  # connect to self
flow = AsyncFlow(start=agent)

# Create heartbeat sender
async def send_system_messages(message_queue):
    counter = 0
    messages = [
        "System status: all systems operational",
        "Memory usage: normal",
        "Network connectivity: stable",
        "Processing load: optimal"
    ]
    
    while True:
        message = f"{messages[counter % len(messages)]} | timestamp_{counter}"
        await message_queue.put(message)
        counter += 1
        await asyncio.sleep(1)

async def main():
    message_queue = asyncio.Queue()
    shared = {}
    flow.set_params({"messages": message_queue})
    
    # Run both coroutines
    await asyncio.gather(
        flow.run_async(shared),
        send_system_messages(message_queue)
    )
    
asyncio.run(main())
```

The output:

```
Agent received: System status: all systems operational | timestamp_0
Agent received: Memory usage: normal | timestamp_1
Agent received: Network connectivity: stable | timestamp_2
Agent received: Processing load: optimal | timestamp_3
```

### Interactive Multi-Agent Example: Taboo Game

Here's a more complex example where two agents play the word-guessing game Taboo. 
One agent provides hints while avoiding forbidden words, and another agent tries to guess the target word:

```python
class AsyncHinter(AsyncNode):
    async def prep_async(self, shared):
        guess = await shared["hinter_queue"].get()
        if guess == "GAME_OVER":
            return None
        return shared["target_word"], shared["forbidden_words"], shared.get("past_guesses", [])

    async def exec_async(self, inputs):
        if inputs is None:
            return None
        target, forbidden, past_guesses = inputs
        prompt = f"Generate hint for '{target}'\nForbidden words: {forbidden}"
        if past_guesses:
            prompt += f"\nPrevious wrong guesses: {past_guesses}\nMake hint more specific."
        prompt += "\nUse at most 5 words."
        
        hint = call_llm(prompt)
        print(f"\nHinter: Here's your hint - {hint}")
        return hint

    async def post_async(self, shared, prep_res, exec_res):
        if exec_res is None:
            return "end"
        await shared["guesser_queue"].put(exec_res)
        return "continue"

class AsyncGuesser(AsyncNode):
    async def prep_async(self, shared):
        hint = await shared["guesser_queue"].get()
        return hint, shared.get("past_guesses", [])

    async def exec_async(self, inputs):
        hint, past_guesses = inputs
        prompt = f"Given hint: {hint}, past wrong guesses: {past_guesses}, make a new guess. Directly reply a single word:"
        guess = call_llm(prompt)
        print(f"Guesser: I guess it's - {guess}")
        return guess

    async def post_async(self, shared, prep_res, exec_res):
        if exec_res.lower() == shared["target_word"].lower():
            print("Game Over - Correct guess!")
            await shared["hinter_queue"].put("GAME_OVER")
            return "end"
            
        if "past_guesses" not in shared:
            shared["past_guesses"] = []
        shared["past_guesses"].append(exec_res)
        
        await shared["hinter_queue"].put(exec_res)
        return "continue"

async def main():
    # Set up game
    shared = {
        "target_word": "nostalgia",
        "forbidden_words": ["memory", "past", "remember", "feeling", "longing"],
        "hinter_queue": asyncio.Queue(),
        "guesser_queue": asyncio.Queue()
    }
    
    print("Game starting!")
    print(f"Target word: {shared['target_word']}")
    print(f"Forbidden words: {shared['forbidden_words']}")

    # Initialize by sending empty guess to hinter
    await shared["hinter_queue"].put("")

    # Create nodes and flows
    hinter = AsyncHinter()
    guesser = AsyncGuesser()

    # Set up flows
    hinter_flow = AsyncFlow(start=hinter)
    guesser_flow = AsyncFlow(start=guesser)

    # Connect nodes to themselves
    hinter - "continue" >> hinter
    guesser - "continue" >> guesser

    # Run both agents concurrently
    await asyncio.gather(
        hinter_flow.run_async(shared),
        guesser_flow.run_async(shared)
    )

asyncio.run(main())
```

The Output:

```
Game starting!
Target word: nostalgia
Forbidden words: ['memory', 'past', 'remember', 'feeling', 'longing']

Hinter: Here's your hint - Thinking of childhood summer days
Guesser: I guess it's - popsicle

Hinter: Here's your hint - When childhood cartoons make you emotional
Guesser: I guess it's - nostalgic

Hinter: Here's your hint - When old songs move you
Guesser: I guess it's - memories

Hinter: Here's your hint - That warm emotion about childhood
Guesser: I guess it's - nostalgia
Game Over - Correct guess!
```

================================================
File: docs/design_pattern/rag.md
================================================
---
layout: default
title: "RAG"
parent: "Design Pattern"
nav_order: 4
---

# RAG (Retrieval Augmented Generation)

For certain LLM tasks like answering questions, providing context is essential.
Use [vector search](../utility_function/tool.md) to find relevant context for LLM responses.

### Example: Question Answering

```python
class PrepareEmbeddings(Node):
    def prep(self, shared):
        return shared["texts"]

    def exec(self, texts):
        # Embed each text chunk
        embs = [get_embedding(t) for t in texts]
        return embs

    def post(self, shared, prep_res, exec_res):
        shared["search_index"] = create_index(exec_res)
        # no action string means "default"

class AnswerQuestion(Node):
    def prep(self, shared):
        question = input("Enter question: ")
        return question

    def exec(self, question):
        q_emb = get_embedding(question)
        idx, _ = search_index(shared["search_index"], q_emb, top_k=1)
        best_id = idx[0][0]
        relevant_text = shared["texts"][best_id]
        prompt = f"Question: {question}\nContext: {relevant_text}\nAnswer:"
        return call_llm(prompt)

    def post(self, shared, p, answer):
        print("Answer:", answer)

############################################
# Wire up the flow
prep = PrepareEmbeddings()
qa = AnswerQuestion()
prep >> qa

flow = Flow(start=prep)

# Example usage
shared = {"texts": ["I love apples", "Cats are great", "The sky is blue"]}
flow.run(shared)
```

================================================
File: docs/design_pattern/structure.md
================================================
---
layout: default
title: "Structured Output"
parent: "Design Pattern"
nav_order: 1
---

# Structured Output

In many use cases, you may want the LLM to output a specific structure, such as a list or a dictionary with predefined keys.

There are several approaches to achieve a structured output:
- **Prompting** the LLM to strictly return a defined structure.
- Using LLMs that natively support **schema enforcement**.
- **Post-processing** the LLM's response to extract structured content.

In practice, **Prompting** is simple and reliable for modern LLMs.

### Example Use Cases

- Extracting Key Information 

```yaml
product:
  name: Widget Pro
  price: 199.99
  description: |
    A high-quality widget designed for professionals.
    Recommended for advanced users.
```

- Summarizing Documents into Bullet Points

```yaml
summary:
  - This product is easy to use.
  - It is cost-effective.
  - Suitable for all skill levels.
```

- Generating Configuration Files

```yaml
server:
  host: 127.0.0.1
  port: 8080
  ssl: true
```

## Prompt Engineering

When prompting the LLM to produce **structured** output:
1. **Wrap** the structure in code fences (e.g., `yaml`).
2. **Validate** that all required fields exist (and let `Node` handles retry).

### Example Text Summarization

```python
class SummarizeNode(Node):
    def exec(self, prep_res):
        # Suppose `prep_res` is the text to summarize.
        prompt = f"""
Please summarize the following text as YAML, with exactly 3 bullet points

{prep_res}

Now, output:
```yaml
summary:
  - bullet 1
  - bullet 2
  - bullet 3
```"""
        response = call_llm(prompt)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()

        import yaml
        structured_result = yaml.safe_load(yaml_str)

        assert "summary" in structured_result
        assert isinstance(structured_result["summary"], list)

        return structured_result
```

> Besides using `assert` statements, another popular way to validate schemas is [Pydantic](https://github.com/pydantic/pydantic)
{: .note }

### Why YAML instead of JSON?

Current LLMs struggle with escaping. YAML is easier with strings since they don't always need quotes.

**In JSON**  

```json
{
  "dialogue": "Alice said: \"Hello Bob.\\nHow are you?\\nI am good.\""
}
```

- Every double quote inside the string must be escaped with `\"`.
- Each newline in the dialogue must be represented as `\n`.

**In YAML**  

```yaml
dialogue: |
  Alice said: "Hello Bob.
  How are you?
  I am good."
```

- No need to escape interior quotes—just place the entire text under a block literal (`|`).
- Newlines are naturally preserved without needing `\n`.

================================================
File: docs/design_pattern/workflow.md
================================================
---
layout: default
title: "Workflow"
parent: "Design Pattern"
nav_order: 2
---

# Workflow

Many real-world tasks are too complex for one LLM call. The solution is to decompose them into a [chain](../core_abstraction/flow.md) of multiple Nodes.

> - You don't want to make each task **too coarse**, because it may be *too complex for one LLM call*.
> - You don't want to make each task **too granular**, because then *the LLM call doesn't have enough context* and results are *not consistent across nodes*.
> 
> You usually need multiple *iterations* to find the *sweet spot*. If the task has too many *edge cases*, consider using [Agents](./agent.md).
{: .best-practice }

### Example: Article Writing

```python
class GenerateOutline(Node):
    def prep(self, shared): return shared["topic"]
    def exec(self, topic): return call_llm(f"Create a detailed outline for an article about {topic}")
    def post(self, shared, prep_res, exec_res): shared["outline"] = exec_res

class WriteSection(Node):
    def prep(self, shared): return shared["outline"]
    def exec(self, outline): return call_llm(f"Write content based on this outline: {outline}")
    def post(self, shared, prep_res, exec_res): shared["draft"] = exec_res

class ReviewAndRefine(Node):
    def prep(self, shared): return shared["draft"]
    def exec(self, draft): return call_llm(f"Review and improve this draft: {draft}")
    def post(self, shared, prep_res, exec_res): shared["final_article"] = exec_res

# Connect nodes
outline = GenerateOutline()
write = WriteSection()
review = ReviewAndRefine()

outline >> write >> review

# Create and run flow
writing_flow = Flow(start=outline)
shared = {"topic": "AI Safety"}
writing_flow.run(shared)
```

For *dynamic cases*, consider using [Agents](./agent.md).

================================================
File: docs/utility_function/llm.md
================================================
---
layout: default
title: "LLM Wrapper"
parent: "Utility Function"
nav_order: 1
---

# LLM Wrappers  

We **don't** provide built-in LLM wrappers. Instead, please implement your own, for example by asking an assistant like ChatGPT or Claude. If you ask ChatGPT to "implement a `call_llm` function that takes a prompt and returns the LLM response," you shall get something like:

```python
def call_llm(prompt):
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content

# Example usage
call_llm("How are you?")
```

> Store the API key in an environment variable like OPENAI_API_KEY for security.
{: .note }

## Improvements
Feel free to enhance your `call_llm` function as needed. Here are examples:

- Handle chat history:

```python
def call_llm(messages):
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return r.choices[0].message.content
```

- Add in-memory caching 

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def call_llm(prompt):
    # Your implementation here
    pass
```

> ⚠️ Caching conflicts with Node retries, as retries yield the same result.
>
> To address this, you could use cached results only if not retried.
{: .warning }


```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_call(prompt):
    pass

def call_llm(prompt, use_cache):
    if use_cache:
        return cached_call(prompt)
    # Call the underlying function directly
    return cached_call.__wrapped__(prompt)

class SummarizeNode(Node):
    def exec(self, text):
        return call_llm(f"Summarize: {text}", self.cur_retry==0)
```

- Enable logging:

```python
def call_llm(prompt):
    import logging
    logging.info(f"Prompt: {prompt}")
    response = ... # Your implementation here
    logging.info(f"Response: {response}")
    return response
```

## Why Not Provide Built-in LLM Wrappers?
I believe it is a **bad practice** to provide LLM-specific implementations in a general framework:
- **LLM APIs change frequently**. Hardcoding them makes maintenance a nightmare.
- You may need **flexibility** to switch vendors, use fine-tuned models, or deploy local LLMs.
- You may need **optimizations** like prompt caching, request batching, or response streaming.

================================================
File: docs/utility_function/tool.md
================================================
---
layout: default
title: "Tool"
parent: "Utility Function"
nav_order: 2
---

# Tool

Similar to LLM wrappers, we **don't** provide built-in tools. Here, we recommend some *minimal* (and incomplete) implementations of commonly used tools. These examples can serve as a starting point for your own tooling.

---

## 1. Embedding Calls

```python
def get_embedding(text):
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    r = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return r.data[0].embedding

get_embedding("What's the meaning of life?")
```

---

## 2. Vector Database (Faiss)

```python
import faiss
import numpy as np

def create_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(index, query_embedding, top_k=5):
    D, I = index.search(
        np.array([query_embedding]).astype('float32'), 
        top_k
    )
    return I, D

index = create_index(embeddings)
search_index(index, query_embedding)
```

---

## 3. Local Database

```python
import sqlite3

def execute_sql(query):
    conn = sqlite3.connect("mydb.db")
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    return result
```

> ⚠️ Beware of SQL injection risk
{: .warning }

---

## 4. Python Function Execution

```python
def run_code(code_str):
    env = {}
    exec(code_str, env)
    return env

run_code("print('Hello, world!')")
```

> ⚠️ exec() is dangerous with untrusted input
{: .warning }


---

## 5. PDF Extraction

If your PDFs are text-based, use PyMuPDF:

```python
import fitz  # PyMuPDF

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

extract_text("document.pdf")
```

For image-based PDFs (e.g., scanned), OCR is needed. A easy and fast option is using an LLM with vision capabilities:

```python
from openai import OpenAI
import base64

def call_llm_vision(prompt, image_data):
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    img_base64 = base64.b64encode(image_data).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }]
    )
    
    return response.choices[0].message.content

pdf_document = fitz.open("document.pdf")
page_num = 0
page = pdf_document[page_num]
pix = page.get_pixmap()
img_data = pix.tobytes("png")

call_llm_vision("Extract text from this image", img_data)
```

---

## 6. Web Crawling

```python
def crawl_web(url):
    import requests
    from bs4 import BeautifulSoup
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.string, soup.get_text()
```

---

## 7. Basic Search (SerpAPI example)

```python
def search_google(query):
    import requests
    params = {
        "engine": "google",
        "q": query,
        "api_key": "YOUR_API_KEY"
    }
    r = requests.get("https://serpapi.com/search", params=params)
    return r.json()
```

---


## 8. Audio Transcription (OpenAI Whisper)

```python
def transcribe_audio(file_path):
    import openai
    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]
```

---

## 9. Text-to-Speech (TTS)

```python
def text_to_speech(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
```

---

## 10. Sending Email

```python
def send_email(to_address, subject, body, from_address, password):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_address, password)
        server.sendmail(from_address, [to_address], msg.as_string())
```

================================================
File: docs/utility_function/viz.md
================================================
---
layout: default
title: "Viz and Debug"
parent: "Utility Function"
nav_order: 3
---

# Visualization and Debugging

Similar to LLM wrappers, we **don't** provide built-in visualization and debugging. Here, we recommend some *minimal* (and incomplete) implementations These examples can serve as a starting point for your own tooling.

## 1. Visualization with Mermaid

This code recursively traverses the nested graph, assigns unique IDs to each node, and treats Flow nodes as subgraphs to generate Mermaid syntax for a hierarchical visualization.

{% raw %}
```python
def build_mermaid(start):
    ids, visited, lines = {}, set(), ["graph LR"]
    ctr = 1
    def get_id(n):
        nonlocal ctr
        return ids[n] if n in ids else (ids.setdefault(n, f"N{ctr}"), (ctr := ctr + 1))[0]
    def link(a, b):
        lines.append(f"    {a} --> {b}")
    def walk(node, parent=None):
        if node in visited:
            return parent and link(parent, get_id(node))
        visited.add(node)
        if isinstance(node, Flow):
            node.start and parent and link(parent, get_id(node.start))
            lines.append(f"\n    subgraph sub_flow_{get_id(node)}[{type(node).__name__}]")
            node.start and walk(node.start)
            for nxt in node.successors.values():
                node.start and walk(nxt, get_id(node.start)) or (parent and link(parent, get_id(nxt))) or walk(nxt)
            lines.append("    end\n")
        else:
            lines.append(f"    {(nid := get_id(node))}['{type(node).__name__}']")
            parent and link(parent, nid)
            [walk(nxt, nid) for nxt in node.successors.values()]
    walk(start)
    return "\n".join(lines)
```
{% endraw %}


For example, suppose we have a complex Flow for data science:

```python
class DataPrepBatchNode(BatchNode):
    def prep(self,shared): return []
class ValidateDataNode(Node): pass
class FeatureExtractionNode(Node): pass
class TrainModelNode(Node): pass
class EvaluateModelNode(Node): pass
class ModelFlow(Flow): pass
class DataScienceFlow(Flow):pass

feature_node = FeatureExtractionNode()
train_node = TrainModelNode()
evaluate_node = EvaluateModelNode()
feature_node >> train_node >> evaluate_node
model_flow = ModelFlow(start=feature_node)
data_prep_node = DataPrepBatchNode()
validate_node = ValidateDataNode()
data_prep_node >> validate_node >> model_flow
data_science_flow = DataScienceFlow(start=data_prep_node)
result = build_mermaid(start=data_science_flow)
```

The code generates a Mermaid diagram:

```mermaid
graph LR
    subgraph sub_flow_N1[DataScienceFlow]
    N2['DataPrepBatchNode']
    N3['ValidateDataNode']
    N2 --> N3
    N3 --> N4

    subgraph sub_flow_N5[ModelFlow]
    N4['FeatureExtractionNode']
    N6['TrainModelNode']
    N4 --> N6
    N7['EvaluateModelNode']
    N6 --> N7
    end

    end
```

## 2. Call Stack Debugging

It would be useful to print the Node call stacks for debugging. This can be achieved by inspecting the runtime call stack:

```python
import inspect

def get_node_call_stack():
    stack = inspect.stack()
    node_names = []
    seen_ids = set()
    for frame_info in stack[1:]:
        local_vars = frame_info.frame.f_locals
        if 'self' in local_vars:
            caller_self = local_vars['self']
            if isinstance(caller_self, BaseNode) and id(caller_self) not in seen_ids:
                seen_ids.add(id(caller_self))
                node_names.append(type(caller_self).__name__)
    return node_names
```

For example, suppose we have a complex Flow for data science:

```python
class DataPrepBatchNode(BatchNode): 
    def prep(self, shared): return []
class ValidateDataNode(Node): pass
class FeatureExtractionNode(Node): pass
class TrainModelNode(Node): pass
class EvaluateModelNode(Node): 
    def prep(self, shared):
        stack = get_node_call_stack()
        print("Call stack:", stack)
class ModelFlow(Flow): pass
class DataScienceFlow(Flow):pass

feature_node = FeatureExtractionNode()
train_node = TrainModelNode()
evaluate_node = EvaluateModelNode()
feature_node >> train_node >> evaluate_node
model_flow = ModelFlow(start=feature_node)
data_prep_node = DataPrepBatchNode()
validate_node = ValidateDataNode()
data_prep_node >> validate_node >> model_flow
data_science_flow = DataScienceFlow(start=data_prep_node)
data_science_flow.run({})
```

The output would be: `Call stack: ['EvaluateModelNode', 'ModelFlow', 'DataScienceFlow']`