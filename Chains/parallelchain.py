

#---------------  USING CHATOPEN AI  ----#

# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel

# load_dotenv()

# model1 = ChatOpenAI()

# model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

# prompt1 = PromptTemplate(
#     template='Generate short and simple notes from the following text \n {text}',
#     input_variables=['text']
# )

# prompt2 = PromptTemplate(
#     template='Generate 5 short question answers from the following text \n {text}',
#     input_variables=['text']
# )

# prompt3 = PromptTemplate(
#     template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
#     input_variables=['notes', 'quiz']
# )

# parser = StrOutputParser()

# parallel_chain = RunnableParallel({
#     'notes': prompt1 | model1 | parser,
#     'quiz': prompt2 | model2 | parser
# })

# merge_chain = prompt3 | model1 | parser

# chain = parallel_chain | merge_chain

# text = """
# Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

# The advantages of support vector machines are:

# Effective in high dimensional spaces.

# Still effective in cases where number of dimensions is greater than the number of samples.

# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# The disadvantages of support vector machines include:

# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
# """

# result = chain.invoke({'text':text})

# print(result)

# chain.get_graph().print_ascii()

#----------------------------------------------------- USING OPENSOURCE MODELS -----------------------------------------------------



from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

# Force model cache to D drive
os.environ['TRANSFORMERS_CACHE'] = "D:/hf_models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load model (choose light but usable one)
model_id = "distilgpt2"  # ~82MB, small but makes readable English
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# creating hugging face pipeline
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
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# Output parser
parser = StrOutputParser()

# Parallel execution
parallel_chain = RunnableParallel({
    'notes': prompt1 | llm | parser,
    'quiz': prompt2 | llm | parser
})

merge_chain = prompt3 | llm | parser
chain = parallel_chain | merge_chain

text = """
gentic AI refers to AI systems designed to act autonomously, making decisions and taking actions to achieve specific goals 
with limited human supervision. These systems are characterized by their ability to understand context, adapt to new information,
 and learn from interactions to improve their performance. Unlike traditional automation, agentic AI systems are not limited to
   predefined tasks and can handle complex challenges by combining different AI techniques and tools
"""

# Run the chain
result = chain.invoke({'text': text})

# Output result
print(result)

# Print graph
chain.get_graph().print_ascii()
