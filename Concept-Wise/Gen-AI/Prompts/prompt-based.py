from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st 


load_dotenv()
model = ChatMistralAI(
    model = "mistral-medium-2508",
)

st.header("Car Dealer-ship Tool")

Car_name = st.selectbox(
    "Select the Car you want to Buy",
    [
        "1. Rolls Royces Phantom",
        "2. Range Rover",
        "3. Supra",
        "4. Mercedez C class" 
    ]
)

Car_color = st.selectbox(
    "Select Your color",
    [
        "Red",
        "Blue",
        "green",
        "sky-grey"
    ]
)

# Designing user-prompt 
chat_templete = ChatPromptTemplate.from_messages(
    [
        ('system',"you are an expert car dealer"),
        ("user","tell everything about {Car_name} with {Car_color}, including max speed, milage,on road price and etc")
    ]
)
# Filling the placeholder 

user_prompt = chat_templete.invoke(
    {
        "Car_name" : Car_name,
        "Car_color" : Car_color
    }
)

if st.button("Show car details"):
    response = model.invoke(user_prompt)
    st.write(response.content)