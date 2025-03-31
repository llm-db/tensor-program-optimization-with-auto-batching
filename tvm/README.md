There are the following folders:
- **custom**: contains custom TVM implementations including the `FuseTakeMatmul` optimization
- **llama_auto_batcher**: contains all code of the Auto-Batching framework
- **lora_only**: contains code for compiling and running just the LoRA computations with TVM (including the Multi-Rank LoRA case study)
- **no_adapters**: contains code for compiling and running the Llama model using TVM
- **opt_configs**: contains the default optimization config file