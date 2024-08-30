import os
import sys
work_dir = os.getcwd()
sys.path.append(work_dir)

from unsloth import FastLanguageModel
from src.fineture_llama3.helper import get_alpaca_prompt
from src.lib.file import get_llama3_model_path
from transformers import TextStreamer

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

from unsloth import FastLanguageModel

#  load the LoRA adapters we just saved for inference,
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

alpaca_prompt = get_alpaca_prompt()
inputs = tokenizer(
[
    alpaca_prompt.format(
        "朱利叶斯·凯撒是怎么死的?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))


# You can also use a TextStreamer for continuous inference -
# so you can see the generation token by token, instead of waiting the whole time!
text_streamer = TextStreamer(tokenizer)
outputs = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
print(tokenizer.batch_decode(outputs))