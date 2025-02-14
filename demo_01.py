from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "deepseek-r1:14b",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)

messages = [
    ("system", "You are a helpful assistant"),
    ("human", "1+1=?"),
]

result = llm.invoke(messages).content

print(result)
