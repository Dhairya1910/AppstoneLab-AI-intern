
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv() # loading Env

model = ChatOpenRouter(
    model="openai/gpt-oss-120b:free",
    temperature = 1.0
)
print("Model loaded Successfully.")

response = model.invoke("Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands. in python")
print(response.content)