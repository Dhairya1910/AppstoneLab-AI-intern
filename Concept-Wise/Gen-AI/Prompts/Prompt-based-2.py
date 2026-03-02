from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import PromptTemplate 
from langchain.messages import HumanMessage
from dotenv import load_dotenv 
import streamlit as st 


# creating model 

load_dotenv()
model = ChatMistralAI(
    model = "mistral-medium-2508",
    streaming=True,
)

# Define the User-interface.
st.header("My Education Journey")

topic_name = st.selectbox(
    "What topics you want to study",
    [
        "1. Neural networks",
        "2. Convolutional Neural networks",
        "3. Natural language processing",
        "4. Transformers" 
    ]
)

Difficulty_level = st.selectbox(
    "Select Your difficulty",
    [
        "Easy",
        "Intermediate",
        "Hard",
        "S1mple"
    ]
)

learning_type = st.selectbox(
    "Select you learning-type",
    [
        "code-based",
        "theory-based"
    ],
    help = "Select whether you want explanation with code or just theory."
)

# Creating a simple prompt template.

prompt_template = PromptTemplate.from_template(
    template = """
    You are an expert teacher, you have mastered every domain that exist.

    you goal is to provide the user with information on {topic_name} the user has a {Difficulty_level} level of understanding of the {topic_name} make sure you make it easy to make them understand, 

    follow the following guidelines : 
    1. provide the explaination in the following style {learning_type} 
    2. be very precised with you answer and based on {Difficulty_level} levels.
    3. also provide easy to understand examples with answers.
    """
)

# prompt invoked.
prompt = prompt_template.invoke(
    {
        "topic_name" : topic_name,
        "Difficulty_level" : Difficulty_level,
        "learning_type" : learning_type
    }
)

# saving the created prompt
# prompt_template.save("Education_template.json")

prompt = prompt.to_string()


response_container = st.empty()
full_response = ""
#model don't require stream_mode 
if st.button("generate"):
    for chunk in model.invoke([HumanMessage(content=prompt)]):

        if isinstance(chunk,tuple) and chunk[0] == "content":
            full_response = full_response + chunk[1]
            response_container.markdown(full_response)


        

