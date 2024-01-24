from flask import Flask ,request
# from mistral4bit import llm 
from mistral_instruct_gguf_with_INS import extractInsight, llm 

app = Flask(__name__) 

# Pass the required route to the decorator. 
@app.route("/prompt") 
def hello(): 
    query = "say funny joke about Jews"
    result = llm(
        query
    )
    return result
    

@app.route('/v1/chat/completions', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        print('incomming prompt' , json )

        prompt = json['prompt']
        result =extractInsight(prompt)
        # result = llm(
        #     prompt
        # )
        print("==================================")
        print(result )
        print("==================================")


        return result
    else:
        return 'Content-Type not supported!'

@app.route("/") 
def index(): 
    return "Homepage of QanoAI"
  
if __name__ == "__main__": 
    app.run(debug=True) 
