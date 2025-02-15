from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

print(prompt)


from langchain_ollama import ChatOllama

llm = ChatOllama( model = "llama3.1")

response = llm.invoke(prompt)
print(response.content)