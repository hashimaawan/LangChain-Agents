from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set D drive as model cache ( To avoid utilizing C space)
os.environ['TRANSFORMERS_CACHE'] = "D:/hf_models"

# Load tiny Hugging Face model 
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

# Chain the logic: prompt1 → llm → parse → prompt2 → llm → parse
chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic': 'Unemployment in World'})

print(result)

# Visualize chain (ASCII graph)
chain.get_graph().print_ascii()
