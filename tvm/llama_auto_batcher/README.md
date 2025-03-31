The implementation of the Auto-Batching framework can be found in the following files:
- **llama_for_causal_lm.py**: contains the Llama model definition
- **lora.py**: contains the LoRA definition
- **exec_graph.py**: contains the base class for representing an execution graph, including the fusion algorithm
- **llama_exec_graph.py**: contains the execution graph class specific to the llama model, with implementations of all operations (using TVM-compiled functions)
- **compile_tvm.py**: contains the function compilation with TVM
- **llama_seq_info.py**: implements a class for storing info like adapter method or generation length for each sequence
- **llama_auto_batcher.py**: the main class that is exposed to the user, with functions to build the graph and `compile` and `execute` functions

There are two ways to use the Auto-Batching framework. **user_example.py** contains an example of how the framework is intended to be used by a user. It can be executed with `python user_example.py` and takes no parameters. **main.py** on the other hand, takes parameters and automatically builds the graph, compiles and executes using the framework's functions. It was used for our measurements. It takes the following parameters, whose default values can be overwritten with `--<parameter_name>=<value>`.
- **opt_config** (fineinfer-autopeft/tvm/opt_configs/default.json): path to `json` file storing which optional TVM optimizations should be applied
- **gen_len** (32)
- **prompt** (fininfer-autopeft/prompts/default.txt): path to text file storing the prompt
- **lora_ranks**: different LoRA ranks
- **batch_size** (1)
- **num_seqs_per_adapter**: number of sequences for each LoRA rank. If the sum is smaller than **batch_size**, then there are also requests that dont use adapters
- **request_type** (uniform): LoRA request distribution within the same LoRA rank (distinct, uniform or identical)
- **measure** (False): whether or not to take latency measurements
- **measure_dest** (fineinfer-autopeft/measure.txt): path to text file where the measurements are written to
- **warmup_trials** (3): number of trials that are executed before the trials that are measured
- **measure_trials** (1): number of trials that are measured
- **fuse_decode** (True): whether or not to fuse the requests for the decode phase
