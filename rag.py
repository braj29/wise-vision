from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
import openai
from openai import OpenAI
from os import getenv
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def setup_retriever():
    knowledge_file = "E:/Backup_K20pro/Download/treesat_benchmark/Canopy species list and uses (PT)_pt_en.docx"
    loader = Docx2txtLoader(knowledge_file) 
    data = loader.load()
    text_splitter = CharacterTextSplitter(
    separator="-",
    chunk_size=600,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
    chunks = text_splitter.split_documents(data)
# print(chunks)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 1
    return retriever

retriever = setup_retriever()
species = 'Byrsonima'
species_info = retriever.invoke(f"Scientific name:{species}")
print(species_info)
# qa_translator = DoctranTextTranslator(language="english")
# translated_document = qa_translator.transform_documents(data)
# print(translated_document)


info = 'Scientific name:Licaniasp.\n\nFamily:Chrysobalanaceae\n\nPopular name:They are generally known as caripé or macucu, among other generic names.\n\nHabitat:LicaniaIt is a large genus, with dozens of species distributed in all Amazon habitats, species that are difficult to identify without fertile material.\n\nUses:Some species can be edible, others known as caripé had (have) their bark roasted (presence of silica), macerated and together with clay used in the preparation of ceramics by indigenous populations.'
# gets API Key from environment variable OPENAI_API_KEY
def setup_client():
    client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)
    
    return client

client = setup_client()

def generate_image(species_info, client):
    completion = client.chat.completions.create(
  model="openai/gpt-3.5-turbo",
  messages=[
    {
      "role": "user",

      "content": f"Using the trees species information provided below, Using the information in the 'Uses:' section. Generate 1 useful and informative unicode image to be used to be placed on a drone panoramic image. Tree species info:{species_info}",
    },
  ],
)
    return completion.choices[0].message.content


if __name__ == '__main__':
    ans = generate_image(species_info, client)
    print(ans)
# # Define the prompt and parameters for the request
# prompt = "Once upon a time"
# response = openai.Completion.create(
#     engine="gpt-3.5-turbo",  # Use the GPT-3.5 model
#     prompt=prompt,
#     max_tokens=50,  # Adjust the number of tokens based on your requirement
#     n=1,
#     stop=None,
#     temperature=0.7,
# )

# # Print the generated text
# generated_text = response.choices[0].text.strip()
# print(generated_text)