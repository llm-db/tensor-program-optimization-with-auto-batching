There are two files for the multi-rank LoRA case. `multiple_ranks.py` implements the sequential, padded and splitted approaches. It supports any number of different ranks, while all ranks must be divisible by the smallest rank, such that the splitted approach works without padding. `two_ranks_with_combine.py` implements the sequential, padded, splitted and combined approaches. It supports only two different ranks. It not only compiles functions for the full LoRA computations but also for LoRA-A and LoRA-B separately. With `python multiple_ranks.py` and `python two_ranks_with_combine.py` they can be executed, and they take different sets of parameters that have default values that can be overwritten with `--<parameter_name>=<value>`.

`multiple_ranks.py` takes the following parameters:
- **opt_config** (fineinfer-autopeft/tvm/opt_configs/default.json): path to `json` file storing which optional TVM optimizations should be applied
- **in_features** (4096): dimension of the input
- **out_features** (4096): dimension of the output
- **request_type** (distinct): LoRA request distribution (distinct, uniform or identical)
- **trials** (1): number of trials
- **measure** (False): whether or not to take latency measurements
- **measure_dest** (fineinfer-autopeft/measure.txt): path to text file where the measurements are written to
- **warmup_trials** (3): number of trials that are executed before the trials that are measured
- **rs** (64,32): different LoRA ranks (separated by comma)
- **batch_sizes** (1,1): the batch size for each LoRA rank (separated by comma)

`two_ranks_with_combine.py` takes the following parameters:
- **request_type** (distinct): LoRA request distribution (distinct, uniform or identical). Combined currently only supports distinct.
- **r_large** (64): the larger of the two LoRA ranks
- **r_small** (32): the smaller of the two LoRA ranks
- **batch_size_large** (1): the batch size for the larger LoRA rank
- **batch_size_small** (2): the batch size for the smaller LoRA rank
- **trials** (1): number of trials
- **measure** (False): whether or not to take latency measurements
- **measure_dest** (fineinfer-autopeft/measure.txt): path to text file where the measurements are written to
- **warmup_trials** (3): number of trials that are executed before the trials that are measured

**in_features** and **out_features** are always set to 4096 in this case. Additionally, we dont have an **opt_config** parameter here, because we always apply all optimizations (to make combined comparable)

