import re


def process_model_name(model_name: str) -> str:
    return (
        re.sub(r"-\d+(B|b)", "", model_name)
        .replace("-2507", "")
        .replace("-A3B", "")
        .replace("-Distill", "")
        .replace("-it", "")
        .replace("-Thinking", "")
        .replace("DeepSeek-", "")
        .replace("-Instruct", "")
        .replace("thinking", "")
        .replace("Qwen3Next", "Qwen3-Next")
        .replace("_", "")
    )
