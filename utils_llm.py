import os
import torch
import openai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from scipy.stats import entropy
import pickle

# from transformers import AutoModelForCausalLM, AutoTokenizer


def get_codemodel(model_name):
    #  device = "cuda"
    if model_name == "llama_2_7b":
        checkpoint = "meta-llama/Llama-2-7b-hf"
        
    elif model_name == "vicuna_7b":
        checkpoint = "lmsys/vicuna-7b-v1.3"
        
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    
    data_file = "data/data_mul_doc_summary.json"
    res_file = "data/answer_file.bin"
    questions = [json.loads(q) for q in open(os.path.expanduser(data_file), "r")]
    responses = {}
    
    for line in tqdm(questions):
        idx_i = line["idx"]
        x = line['x']
        y = line['y']
        
        # Tokenize the input prompt
        inputs = tokenizer(y, return_tensors="pt").to(device)
        outputs = model.forward(
            inputs.input_ids, 
            output_hidden_states=True,
        )
        
        # hidden states
        hidden_layer = -1
        hidden_states = outputs['hidden_states'][hidden_layer][0]
        
        # logit
        logits = outputs['logits']
        shifted_input_ids = inputs.input_ids[:, 1:]   # 1:len
        shifted_logits = logits[:, 0:-1, :]  # 0:len-1
        
        # Convert logits to probabilities
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        gathered_log_probs_1 = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        gathered_log_probs = gathered_log_probs_1.detach().cpu().numpy()
        
        # Convert logits to entropies
        probs = torch.softmax(shifted_logits, dim=-1)[0]
        probs = probs.detach().cpu().numpy()
        entropies = 2 ** (entropy(probs, base=2, axis=-1))
        
        
        tokens = []
        token_logprobs = []
        token_entropies = []
        tokens_idx = []
        token_logprob_entro = []
        for t in range(shifted_input_ids.shape[1]):
            gen_tok_id = shifted_input_ids[:, t]
            gen_tok = tokenizer.decode(gen_tok_id)
            lp = gathered_log_probs[:, t][0]
            entro = entropies[t]

            tokens.append(gen_tok)
            token_logprobs.append(lp)
            token_entropies.append(entro)
            token_logprob_entro.append([gen_tok, lp, entro])
            tokens_idx.append(gen_tok_id.detach().cpu().numpy().tolist())
        
        hidden_states = hidden_states[-1].detach().cpu().to(torch.float32).numpy().tolist()
        
        output = {"question_id": idx_i,
                'x': x,
                'y': y,
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "token_entropies":token_entropies,
                "tokens_idx": tokens_idx,
                "token_logprob_entro": token_logprob_entro,
                "hidden_states": hidden_states,
                }
        
        responses[idx_i] = output
        
    with open(res_file, 'wb') as file:
        pickle.dump(responses, file)
        

    return tokenizer, model


    
if __name__ == "__main__":
    get_codemodel('llama_2_7b')
    
    
    
    

