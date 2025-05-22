from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

app = FastAPI()

class EmailRequest(BaseModel):
    tone: str
    ai_model: str
    language: str
    context: str

def generate_email_gemini(language: str, tone: str, context: str):

    prompt = f"""
    Generate an email in {language} with a  {tone} tone. Context: {context}
    Return the response in this format:
    Subject: //subject
    Body: //body
"""
    response = model.invoke(prompt)
    response_text = response.content.strip()

    subject, body = response_text.split("Body:", 1) if "Body:" in response_text else ("No Subject", response_text)
    subject = subject.replace("Subject: ", "")
    body = body.strip()

    return {"subject": subject, "body": body}

def generate_email_graph(ai_model: str, tone: str, language: str, context: str):
    def email_generation_fn(state):
        if ai_model == "gemini":
            email = generate_email_gemini(language, tone, context)
        else:
            email = "Invalid AI model selected!"
        return {"email": email}
    
    graph = StateGraph(dict)
    graph.add_node("generate_email", email_generation_fn)
    graph.set_entry_point("generate_email")

    return graph.compile()

@app.get("/")
def read_root():
    return {"message": "Hello, AutoComposeBackend is live!"}

@app.post("/generate_email")
async def generate_email(request: EmailRequest):
    graph = generate_email_graph(request.ai_model, request.tone, request.language, request.context)
    response = graph.invoke({})
    return response

