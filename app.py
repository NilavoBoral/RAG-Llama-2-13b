import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
import time
from datasets import load_dataset
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import Concatenate
from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import gradio as gr

# Define the model from Hugging Face
model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()


# Define the tokenizer from Hugging Face
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)


# get API key from app.pinecone.io and environment from console
pinecone.init(
    environment="gcp-starter",
    api_key="a7dddfc1-8eb3-477e-bc69-0b52f0ee201a"
)

index_name = 'rag-llama-2-paper'
index = pinecone.Index(index_name)

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

text_field = 'text'  # field in metadata that contains text content
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)

# Function to generate text using the model
def answer(Question):
    return rag_pipeline(Question)['result']


# Create a Gradio interface
iface = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(Question="Ask your query"),
    outputs=gr.Textbox(),
    title="Know Llama-2",
    description="Ask the Llama-2-13b model anything about itself.",
)

# Launch the Gradio app
iface.launch()