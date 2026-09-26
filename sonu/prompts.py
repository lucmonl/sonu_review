"""Prompt construction and response-only masking for the Section 3.1 tasks."""

import json

_SYSTEM_TEMPLATE = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions.\n\n"
    "Here are the available tools:\n"
    "<tools>\n"
    "{tools}\n"
    "</tools>\n\n"
    "For each function call, return a JSON object with the function name and "
    "arguments within <tool_call></tool_call> XML tags as follows:\n"
    "<tool_call>\n"
    '{{"name": "function_name", "arguments": {{...}}}}\n'
    "</tool_call>"
)


def _build_prompt_and_response(example):
    """
    Build the prompt string (system + user turns, open assistant header) and
    the response string (assistant tool-call content + eot).

    Mirrors the token sequence that apply_chat_template(messages, tools=tools,
    add_generation_prompt=True, enable_thinking=False) would emit for a
    Llama-3.1/3.2-Instruct tokenizer.
    """
    tools = json.loads(example["tools"])
    tools_str = json.dumps(tools, indent=2)
    system_content = _SYSTEM_TEMPLATE.format(tools=tools_str)
    query = example["query"]
    answers = json.loads(example["answers"])
    response_parts = [
        f"<tool_call>\n{json.dumps(call)}\n</tool_call>" for call in answers
    ]
    response_body = "\n".join(response_parts)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    response = f"{response_body}<|eot_id|>"
    return (prompt, response)


def instruction_prompt_and_response(example):
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + example["instruction"]
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt, example["response"] + "<|eot_id|>"


def tokenize_pair(prompt, response, tokenizer, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels)
