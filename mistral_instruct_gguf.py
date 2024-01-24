# Use a pipeline as a high-level helper
from ctransformers import AutoModelForCausalLM ,LLM ,AutoConfig  , AutoTokenizer 
from ctransformers.transformers import CTransformersModel

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import  pipeline ,GenerationConfig 

MODEL_NAME = "models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"



config = AutoConfig.from_pretrained(model_path_or_repo_id=MODEL_NAME,max_new_tokens=2048 , repetition_penalty=1.1)

# config.max_seq_len=4000
# config.max_answer_len=3000

config.config.max_new_tokens = 4048
config.config.temperature = 0.8
config.config.top_k = 40
config.config.top_p = 0.95
config.config.gpu_layers= 250
config.max_answer_len = 90000
config.max_seq_len = 90000
config.config.max_answer_len = 90000

config.config.context_length = -1

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model:CTransformersModel = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=MODEL_NAME, model_type="mistral" , config=config  , local_files_only=True)

# print(llm("AI is going to"))



# tokenizer = AutoTokenizer.from_pretrained(model )

# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# generation_config.max_new_tokens = 500
# generation_config.temperature = 0.8

# generation_config.top_p = 0.95
# generation_config.top_k = 40

# generation_config.do_sample = True
# generation_config.repetition_penalty = 1.1

pipeline = pipeline(
    "text-generation",
    model=model,
    # tokenizer=tokenizer,
    return_full_text=True,
    # generation_config=config,
)
llm = HuggingFacePipeline(
    pipeline=pipeline,
    )
