from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import PromptTemplate,load_prompt
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
prompt_template = load_prompt("Concept-Wise/Gen-AI/Prompts/Education_template.json")

#model don't require stream_mode 
if st.button("generate"):
    # creating and invoking a chain.
    chain = prompt_template | model 
    response = chain.invoke(
        {
        "topic_name" : topic_name,
        "Difficulty_level" : Difficulty_level,
        "learning_type" : learning_type
        }
    )
    st.write(response.content)




        

