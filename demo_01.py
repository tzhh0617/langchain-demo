from langchain_ollama import ChatOllama

llm = ChatOllama( model = "llama3.1")

messages = [
    ("system", "You are a helpful assistant"),
    ("human", "1+1=?"),
]

result = llm.invoke(messages).content

print(result)
