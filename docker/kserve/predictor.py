from kserve import Model, model_server
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GPT2Model(Model):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/models/distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("/mnt/models/distilgpt2")
        self.model.eval()

    def predict(self, payload):
        input_text = payload["instances"][0]["text"]
        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                max_length=100,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {"predictions": [generated_text]}

if __name__ == "__main__":
    model = GPT2Model("distilgpt2")
    model.load()
    model_server.start([model])
