
def get_alpaca_prompt():
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    {}
    
    ### Input:
    {}
    
    ### Response:
    {}"""

    return alpaca_prompt

def save_model(model, tokenizer):
    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")

    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

def load_model(llama3_model_path, max_seq_length, dtype, load_in_4bit):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=llama3_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    return  model, tokenizer
