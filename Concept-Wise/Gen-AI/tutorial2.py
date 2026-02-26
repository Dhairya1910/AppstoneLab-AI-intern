from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(
    model = 'openai/gpt-oss-120b:free',
)

print('Model loaded successfully.')

agent = create_agent(model = model)
print("Agent Created Successfully.")

for token,metadata in agent.stream(
    {"messages":[HumanMessage(content="who is current Prime minister of India.")]},
    stream_mode = "messages"
    ):
    if token.content:
        print(token.content,end="",flush=True)
