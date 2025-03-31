from llama_auto_batcher import LlamaAutoBatcher
from lora import LoraConfig

def main():
  prompt = open(f"../../prompts/default.txt", "r").read()

  input_var = LlamaAutoBatcher.create_var(prompt)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=10)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=64, alpha=32), wid=0)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=30)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=64, alpha=32), wid=0)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=30)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=64, alpha=32), wid=1)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=30)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=64, alpha=32), wid=2)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=30)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=32, alpha=16), wid=0)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=20)
  LlamaAutoBatcher.print(result)

  input_var = LlamaAutoBatcher.create_var(prompt)
  LlamaAutoBatcher.set_adapter(LoraConfig(r=32, alpha=16), wid=1)
  result = LlamaAutoBatcher.generate(input_var, max_gen_len=20)
  LlamaAutoBatcher.print(result)

  LlamaAutoBatcher.compile()
  LlamaAutoBatcher.execute()

if __name__ == "__main__":
  main()
