from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model from mounted volume
MODEL_PATH = "/mnt/output/"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, prompt: str = Form(...)):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=300,  # Limit the length of generated text
            pad_token_id=tokenizer.pad_token_id,  # Ensure padding token is used correctly
            eos_token_id=tokenizer.eos_token_id,  # Stop generation at EOS token
            num_return_sequences=1,  # You can generate multiple outputs if needed
            no_repeat_ngram_size=4,  # Prevent n-gram repetition
            temperature=0.55,  # Set temperature for more diversity (0.7 is a good middle ground)
            top_p=0.80,  # Use nucleus sampling (top-p) to restrict the sample space
            top_k=40,  # Use top-k sampling (optional, can be removed if not needed)
            do_sample=True  # Enable sampling to use temperature and top_p
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Manually remove the prompt part from the generated text
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    return templates.TemplateResponse("index.html", {"request": request, "output": generated})
