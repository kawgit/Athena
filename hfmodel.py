from transformers import AutoModelForCausalLM
from settings import hfmodel_name

def load_hfmodel():
    return AutoModelForCausalLM.from_pretrained(
        hfmodel_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
