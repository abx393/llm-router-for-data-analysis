from autogen import LLMConfig

gemma = LLMConfig(
    config_list = {
        "api_type": "ollama",
        "model": "gemma3:1b",
    }
)

llama_instruct = LLMConfig(
    config_list = {
        "api_type": "ollama",
        "model": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
         "native_tool_calls": True
    }
)

llama_instruct_3b = LLMConfig(
    config_list={
        "api_type": "ollama",
        "model": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        "native_tool_calls": True
    }
)

qwen = LLMConfig(
    config_list = {
        "api_type": "ollama",
        "model": "qwen3:4b",
        "native_tool_calls": True
    }
)
