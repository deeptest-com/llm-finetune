from unsloth import FastLanguageModel
from src.fineture_llama3.helper import get_alpaca_prompt
from src.lib.file import get_llama3_model_path
from transformers import TextStreamer

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

llama3_model_path = get_llama3_model_path()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = llama3_model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# Try 2x faster inference in a free Colab for Llama-3.1 8b Instruct
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
alpaca_prompt = get_alpaca_prompt()

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
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