from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.output_parsers import StrOutputParser
import time

# from langchain_core.tools import tool
# from Ocr_processor import Fetch_document
from dotenv import load_dotenv

load_dotenv()

# perform OCR and store it into local file.
# print("Fetching Documents")
# file_path = Fetch_document(filepath="Test_doc.pdf")

file_path = r'd:\AppstoneLab-AI-intern\Concept-Wise\Gen-AI\Paper-Checker\Retrived_text\output.txt'

# created loader
loader = TextLoader(file_path=file_path)

# Created Document 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 3000,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(docs)
# print("Chunk summary")
# print(f"Number of chunks formed : {len(chunks)}")
# print(f"Chunk-1 content : \n {chunks[0]}")


prompt_1 = PromptTemplate(
    template = """
    You are an expert Teacher, you are given a summary {summary} of Student exam answersheet accordingly give the student marks and grade.

    Rules: 
    - be somewhat Lenient,
    - return mistakes of student pointwise in the end based on given {summary}.
    - return marks out of 50 and grade of the student.    
    - The test was about Theory of Computation subject.
    - if you cannot evaluate student based on provided summary just say i cannot evaluate the student.
    """
)

prompt_2 = PromptTemplate(
    template = """
    for the given chunk {chunk} generate a summary based on the provided chunk which should contain the student's specific responses to the questions. 

    Rules : 
    - do not add anything extra just a summary of exactly what you read.
    - make sure you skip no questions from the chunks.
    - make the summary in such a way that teacher can evaluate the student using your summary.
    """
)


model = ChatNVIDIA(
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    temperature = 0
)

print("API verified and Model created successfully.")

thinking_model = model.with_thinking_mode(enabled=True)

evaluation_model = ChatMistralAI(
    model = "mistral-medium-2508",
    temperature = 0.5
)

parser = StrOutputParser()
Summary = []

for chunk in chunks:
    chain = prompt_2 | thinking_model | parser
    response = chain.invoke(
        {
            "chunk" : chunk
        }
    )

    time.sleep(1)
    Summary.append(response)


print("Printing the Summary")
for point in Summary:
    print("-"*50)
    print(point)
    print("-"*50)

final_chain = prompt_1 | evaluation_model | parser 
response = final_chain.invoke(
    {
        "summary" : Summary
    }
)

print("*"*50,"Student Evaluation Report","*"*50)
print(response)
    









