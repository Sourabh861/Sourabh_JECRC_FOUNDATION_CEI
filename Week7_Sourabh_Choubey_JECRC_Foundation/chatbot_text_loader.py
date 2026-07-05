from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# load the text file
loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()
text_content = docs[0].page_content

# system message now contains the file content as context
chat_history = [
    SystemMessage(content=f"""You are a helpful assistant. Answer the user's questions 
only based on the following text. If the answer is not in the text, say you don't know.

Text:
{text_content}
""")
]

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:', result.content)