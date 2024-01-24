# Use a pipeline as a high-level helper
# from ctransformers import AutoModelForCausalLM ,LLM ,AutoConfig  , AutoTokenizer 
from langchain_community.llms import CTransformers

MODEL_NAME = "models\models\second-state\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.2.Q4_K_S.gguf"

# , 'max_new_tokens': 4000
config = {'gpu_layers':50, 'context_length':3000, 'temperature':0.8 ,'max_new_tokens':500}
llm = CTransformers(model=MODEL_NAME, model_type="mistral" , local_files_only=True, config = config)
# llm = ctransformers.CTransformers(model=MODEL_NAME, model_type="mistral" , local_files_only=True, gpu_layers=50, context_length = 8192, max_new_tokens= 4096, repetition_penalty = 1.1, temperature=0.8)

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"\n{message['content']}\n\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt

# result = llm(prompt , stop=["</s>"])

def extractInsight(review):
      
  # review = """

  # I am very pleased with all the operations I had, a very caring team, they resolved all my questions and concerns before and after, they provided all kinds of support, they were very understanding, and I don't even want to mention the doctors, they are all very experienced and the best at their job, there was nothing that I was not satisfied with, frankly, even the wrong practices that were done before. They were very good at troubleshooting and gave me correct information, they helped me make the right decisions by acting objectively, not like a commercial enterprise. From now on, it is the only place I will go to with closed eyes for every application 🙌🏼"""
  query = """
  From now on, anytime I send you a "====", you will be given a comment after that "====" where you\'d have to extract all medical related insight tags 
  from the given categories "technology", "facility", "staff", "punctuality", "competence", "cost", and "communication" as you would handle google hospital reviews. Give a score from 1-5 on how positive or negative it is (1 or 2 for  negatvie and 4 or 5 for positive and 3 for neutral). You might also face grammatical, vocabulary and punctuation mistakes, so 
  be sure to fix them before proceeding. The output should be a JSON array of objects enclosed in square brackets ("[" and "]") following the template: {"keyword": <keyword>, "insight": <insight>, "type": "positive or negative", "rating": 1-5, "category": <category>}. For the keyword parameter, select the exact text from the sentence that is the result of your output for the category. If you are not able to process the prompt then say "message cannot be processed." You are not allowed to respond in any way at all except what I asked 
  unless I tell you "stop processing everything." Make sure to double check the array and parse it to see if it works, then send it to me. We start now.\n\n\n===="""+review

  messages = [
    {
      "role": "system",
      "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. The result is always a JSON Array containing Objects. Only 10 Objects Max in the JSON Array and if the review is less than 100 character the result must be atmost 3 Objects."
    },
    {
      "role": "user",
      "content": "From now on, anytime I send you a \"====\", you will be given a comment after that \"====\" where you\\'d have to extract all medical related insight tags \nfrom the given categories \"technology\", \"facility\", \"staff\", \"punctuality\", \"competence\", \"cost\", and \"communication\" as you would handle google hospital reviews. Give a score from 1-5 on how positive or negative it is (1 or 2 for  negatvie and 4 or 5 for positive and 3 for neutral). You might also face grammatical, vocabulary and punctuation mistakes, so \nbe sure to fix them before proceeding. The output should be a JSON array of objects enclosed in square brackets (\"[\" and \"]\") following the template: {\"keyword\": <keyword>, \"insight\": <insight>, \"type\": \"positive or negative\", \"rating\": 1-5, \"category\": <category>}. For the keyword parameter, select the exact text from the sentence that is the result of your output for the category. If you are not able to process the prompt then say \"message cannot be processed.\" You are not allowed to respond in any way at all except what I asked \nunless I tell you \"stop processing everything.\" Make sure to double check the array and parse it to see if it works, then send it to me. We start now.\\n\\n\\n====Look back in slim and fit \\n\\n I found Prettislim to be the best for my weight reduction .It was a satisfied session and the results were seen after my 8 session .I almost lost 30-40 cms and 3.5kgs \nin a month. The doctors and dieticians are friendly and use to always motivate me for the further diet and sessions. Dr Nikita was very sweet and motivated me throughout my session"
    },
    {
      "role": "assistant",
      "content": " [{\"keyword\": \"Prettislim\", \"insight\": \"Effective weight reduction program\", \"type\": \"positive\", \"rating\": 5, \"category\": \"facility\"},\n {\"keyword\": \"satisfied session\", \"insight\": \"Positive patient experience\", \"type\": \"positive\", \"rating\": 5, \"category\": \"punctuality\"},\n {\"keyword\": \"30-40 cms and 3.5kgs\", \"insight\": \"Significant weight loss achievement\", \"type\": \"positive\", \"rating\": 5, \"category\": \"technology\"},\n {\"keyword\": \"doctors and dieticians\", \"insight\": \"Friendly and motivating staff\", \"type\": \"positive\", \"rating\": 5, \"category\": \"staff\"},\n {\"keyword\": \"Dr Nikita\", \"insight\": \"Motivational doctor\", \"type\": \"positive\", \"rating\": 5, \"category\": \"staff\"}]"
    },
  {"role": "user", "content": query}
  ]

  prompt = build_llama2_prompt(messages)

  result = llm(prompt , stop=["</s>"])

  return result
# print(result)