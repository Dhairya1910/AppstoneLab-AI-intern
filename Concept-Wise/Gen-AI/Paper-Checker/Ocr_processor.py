from langchain_nvidia_ai_endpoints import ChatNVIDIA 
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent 


import base64 
import os 
from io import BytesIO
from dotenv import load_dotenv
from pdf2image import convert_from_path
import time 
from PIL import Image

load_dotenv()

def merge_grid(images, rows=2, cols=2):
    """
    Merge images into a grid layout.
    
    images : list of PIL images
    rows   : number of rows
    cols   : number of columns
    """

    # Ensure we only take required number
    images = images[:rows * cols]

    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    max_height = max(heights)

    grid_width = cols * max_width
    grid_height = rows * max_height

    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * max_width
        y = row * max_height

        grid_image.paste(img, (x, y))

    return grid_image

def encode_pdf(filepath):
    """
    Convert PDF pages into images and encode each image to base64.
    """

    images = convert_from_path(
        filepath,
        poppler_path=r"C:\poppler-25.12.0\Library\bin"
    )

    print(f"Total number of image in the pdf : {len(images)}")

    # created an empty list to store vertical stack images into a single chunk in a list.
    merged_images = []
    for i in range(0,len(images),4):
        chunk = images[i:i+4]
        merged = merge_grid(chunk)
        merged_images.append(merged)

    # created an empty list to store encoded vertically stacked images.
    encoded_images = []
    for img in merged_images:
        buffer = BytesIO()
        img.thumbnail((2048,2048))
        img.save(buffer, format="PNG")  
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_images.append(encoded)

    print(f"Total number of images in pdf after processing : {len(encoded_images)}")
    return encoded_images



def Store_res(content):

    folder = r"D:\AppstoneLab-AI-intern\Concept-Wise\Gen-AI\Paper-Checker\Retrived_text"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "output.txt")
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(content + "\n")
    print("Content stored successfully.")
    return filepath
    


def Fetch_document(filepath):

    """
    This function reads a single Document, processes it and stores it content into output.txt 
    
    Args : 
    filepath : path of document you want to process.
    """

    pdf_base64 = encode_pdf(filepath=filepath)
    system_prompt = """
    You are an expert handwriting transcription system.

    Read the handwritten text carefully and transcribe it exactly.

    Rules:
    - Preserve line breaks
    - Do not summarize
    - Do not guess missing words
    - If a word is unreadable write [illegible]

    Return only the transcription.
    """


    ocr_model = ChatNVIDIA(
        model = "meta/llama-4-maverick-17b-128e-instruct",
        temperature = 0,
        max_completion_tokens = 10000,

    )

    Ocr_agent = create_agent(
        model = ocr_model,
        system_prompt = system_prompt,
    )

    start_time = time.perf_counter()
    results = []

    for i in range(0, len(pdf_base64), 8):
        chunk = pdf_base64[i:i+8]

        content = [{"type": "text", "text": "Extract text from these pages. Each page is separated in the grid layout. Preserve page order."}]

        for image in chunk:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            })

        response = Ocr_agent.invoke({
            "messages": [HumanMessage(content=content)]
        })

        results.append(response["messages"][-1].content)

    output = "\n\n".join(results)

    end_time = time.perf_counter()
    print(output)
    print("\n","#"*10,"Time taken to complete : ",(end_time-start_time),"s","#"*10,"\n")

    # Storing the final result in output.txt document
    output_path = Store_res(output)
    return output_path



