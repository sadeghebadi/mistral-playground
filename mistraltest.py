# Use a pipeline as a high-level helper
from langchain_community.llms import CTransformers

# MODEL_NAME = "models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# config = {'gpu_layers':50, 'context_length':16000, 'max_new_tokens': 16000, 'repetition_penalty' : 1.1, 'temperature':0.8}
# llm = CTransformers(model=MODEL_NAME, model_type="mistral" , local_files_only=True, config = config)
# llm = ctransformers.CTransformers(model=MODEL_NAME, model_type="mistral" , local_files_only=True, gpu_layers=50, context_length = 8192, max_new_tokens= 4096, repetition_penalty = 1.1, temperature=0.8)

# result = llm("""
# From now on, anytime I send you a "====", you will be given a comment after that "====" where you\'d have to extract all medical related insight tags 
# from the given categories "technology", "facility", "staff", "punctuality", "competence", "cost", and "communication" as you would handle google hospital reviews. Give a score from 1-5 on how positive or negative it is (1 or 2 for  negatvie and 4 or 5 for positive and 3 for neutral). You might also face grammatical, vocabulary and punctuation mistakes, so 
# be sure to fix them before proceeding. The output should be a JSON array of objects enclosed in square brackets ("[" and "]") following the template: {"keyword": <keyword>, "insight": <insight>, "type": "positive or negative", "rating": 1-5, "category": <category>}. For the keyword parameter, select the exact text from the sentence that is the result of your output for the category. If you are not able to process the prompt then say "message cannot be processed." You are not allowed to respond in any way at all except what I asked 
# unless I tell you "stop processing everything." Make sure to double check the array and parse it to see if it works, then send it to me. We start now.\n\n\n====Look back in slim and fit \n\n I found Prettislim to be the best for my weight reduction .It was a satisfied session and the results were seen after my 8 session .I almost lost 30-40 cms and 3.5kgs 
# in a month. The doctors and dieticians are friendly and use to always motivate me for the further diet and sessions. Dr Nikita was very sweet and motivated me throughout my session
#              """)

# print(result)
print("hello")