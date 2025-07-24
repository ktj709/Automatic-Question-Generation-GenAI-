from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

response = llm.invoke("What is LangChain?")
print(response)
