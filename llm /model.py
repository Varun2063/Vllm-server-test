import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_cache = {}


def load_model(model_name):
    """Load (and cache) a tokenizer/model pair by name."""
    if model_name not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        _cache[model_name] = (tokenizer, model)
    return _cache[model_name]


def generate_answer(
    model_name, prompt, max_new_tokens=32768, temperature=0.7, enable_thinking=True
):
    """Return the generated thinking content and answer as separate fields."""
    tokenizer, model = load_model(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:], skip_special_tokens=True
    ).strip("\n")
    return {"thinking_content": thinking_content, "content": content}


if __name__ == "__main__":
    model_name = input("Model name [Qwen/Qwen3-0.6B]: ").strip() or "Qwen/Qwen3-0.6B"
    question = input("Question: ").strip()
    result = generate_answer(model_name, question)
    print("thinking content:", result["thinking_content"])
    print("content:", result["content"])
