The `custom_peft` folder holds the `GatherBMM` LoRA implementation. 

The `generation` folder contains two generation files, one with LoRA and one without. They can be executed with `python gen.py` and `python gen_lora.py`. There are parameters that can be set with `--<param_name>=<value>`. For generation without LoRA there are the following parameters (with their default values): 
- **model_name** (meta-llama/Meta-Llama-3.1-8B)
- **measure** (False): whether or not to take latency measurements
- **measure_dest** (fineinfer-autopeft/measure.txt): path to text file where the measurements are written to
- **warmup_trials** (3): number of trials that are executed before the trials that are measured
- **trials** (1): number of trials
- **batch_size** (1)
- **gen_len** (32)
- **cache_dir** (/scratch/\<user>): where model weights are cached
- **prompt** (fininfer-autopeft/prompts/default.txt): path to text file storing the prompt
- **attn_impl** (flash_attention_2)

For generation with LoRA there are the following parameters (with their default values):
- **model_name** (meta-llama/Meta-Llama-3.1-8B)
- **measure_full** (False): whether or not to take latency measurements of the full prefill and decode phases
- **measure_adapter** (False): whether or not to take latency measurements of the LoRA adapter computations
- **measure_dest** (fineinfer-autopeft/measure.txt): path to text file where the measurements are written to
- **warmup_trials** (3): number of trials that are executed before the trials that are measured
- **trials** (1): number of trials
- **batch_size** (1)
- **request_type** (uniform): LoRA request distribution (distinct, uniform or identical)
- **gather_bmm** (False): whether or not to use the GatherBMM approach (sequential approach is used otherwise)
- **gen_len** (32)
- **cache_dir** (/scratch/\<user>): where model weights are cached
- **prompt** (fininfer-autopeft/prompts/default.txt): path to text file storing the prompt
- **attn_impl** (flash_attention_2)
