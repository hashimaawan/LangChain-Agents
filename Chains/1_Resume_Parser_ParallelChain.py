import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

# Use D: drive for cache if needed
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model_id = "facebook/bart-large-cnn"   # USE any model that supports summarization
# You can also use "google/flan-t5-large" or similar models for summarization
# Also closed source

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False
)


llm = HuggingFacePipeline(pipeline=pipe)


prompt1 = PromptTemplate(
    template='Summarize the resume and extract the key skills:\n\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 question-answers for interview based on this resume:\n\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Suggest improvements to the resume content:\n\n{text}',
    input_variables=['text']
)

prompt4 = PromptTemplate(
    template='Write a professional summary paragraph based on this resume:\n\n{text}',
    input_variables=['text']
)

prompt5 = PromptTemplate(
    template='Write a cover letter based on this resume:\n\n{text}',
    input_variables=['text']
)


parser = StrOutputParser()

# Parallel chain: run all 5 tasks at once
parallel_chain = RunnableParallel({
    "skills": prompt1 | llm | parser,
    "quiz": prompt2 | llm | parser,
    "improvements": prompt3 | llm | parser,
    "summary": prompt4 | llm | parser,
    "cover_letter": prompt5 | llm | parser
})

# Final merging prompt
merge_prompt = PromptTemplate(
    template="""
Resume Enhancement Report:
 Summary:
{summary}
 Key Skills:
{skills}
Interview Q&A:
{quiz}
Suggested Improvements:
{improvements}
 Suggested Cover Letter:
{cover_letter}
""",
    input_variables=["summary", "skills", "quiz", "improvements", "cover_letter"]
)


final_chain = parallel_chain | merge_prompt | llm | parser

# Example resume text input
resume_text = """
Experienced Software Developer with expertise in Python, Django, and cloud infrastructure.
Managed CI/CD pipelines, Dockerized microservices, and deployed scalable APIs to AWS.
Led a team of 4 developers in agile environments, focused on writing clean, testable code.
"""


result = final_chain.invoke({"text": resume_text})


print(result)


final_chain.get_graph().print_ascii()
