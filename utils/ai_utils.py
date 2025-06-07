from langchain.chat_models import ChatOpenAI

def create_openai_llm(api_key: str):
    """Create a ChatOpenAI LLM object using GPT-4o."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=api_key,
        temperature=0.4
    )
    return llm
