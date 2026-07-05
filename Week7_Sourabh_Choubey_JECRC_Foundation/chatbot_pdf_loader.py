from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# load the PDF file
loader = PyPDFLoader('environment.pdf')
docs = loader.load()

print(len(docs))  # number of pages

# combine content from ALL pages into one string
text_content = "\n\n".join(doc.page_content for doc in docs)

# system message now contains the full PDF content as context
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