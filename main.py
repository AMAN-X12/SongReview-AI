import re
import warnings
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
DEVICE = torch.device("cpu")
bert_model.to(DEVICE)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def aggregate_embeddings(input_ids, attention_masks, bert_model=bert_model):

    add_mean_emb = []

    for token_ids, attention_mask in tqdm(zip(input_ids, attention_masks)):
        token_ids_tensor = torch.tensor([token_ids]).to(DEVICE)
        attention_mask_tensor = torch.tensor([attention_mask]).to(DEVICE)

        with torch.no_grad():

            embeddings = bert_model(token_ids_tensor, attention_mask_tensor)[0].squeeze(0)

        valid_embeddings_mask = attention_mask_tensor[0] != 0

        filtered_embeddings = embeddings[valid_embeddings_mask, :]
        mean_embeddings = filtered_embeddings.mean(0)

        add_mean_emb.append(mean_embeddings.unsqueeze(0))

    aggregated_mean_embeddings = torch.cat(add_mean_emb)
    return aggregated_mean_embeddings

def text_to_emb(list_of_text, max_input=512):
    data_token_index = tokenizer.batch_encode_plus(
        list_of_text,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_input
    )
    question_embeddings = aggregate_embeddings(data_token_index['input_ids'], data_token_index['attention_mask'])
    return question_embeddings



song_questions = [
    "Does this song contain any violent themes, such as references to guns, killing, or physical aggression?",
    "Are there any explicit lyrics or bad words used in this song that might be considered offensive?",
    "Is the overall content of this song suitable for children, considering its themes and language?",
    "Does this song explicitly mention weapons, such as guns, knives, or other similar items?",
    "Are the messages conveyed in this song positive and uplifting for children?",
    "Does this song include any sexual content, references to sexual behavior, or suggestive language?",
    "Does this song offer any educational value, such as teaching the alphabet or basic math?",
    "Does this song promote emotional resilience and social skills among children?"
]

yes_responses = [
    "Yes, this song contains violent themes, including references to guns and killing.",
    "Yes, this song includes explicit lyrics or bad words considered offensive.",
    "No, the overall content of this song is not suitable for children.",
    "Yes, this song explicitly mentions weapons, such as guns and knives.",
    "Yes, the messages conveyed in this song are positive and uplifting.",
    "Yes, this song includes sexual content or suggestive language.",
    "Yes, this song offers significant educational value (ABCs, math, etc.).",
    "Yes, this song promotes emotional resilience and social skills."
]


embeddings_questions = text_to_emb(song_questions)



class songInput(BaseModel):
     lyrics:str
@app.post("/analyze")
async def analyze_song(data:songInput):
    song_new = re.sub(r'[\n]', ' ', data.lyrics)
    song_new = song_new.replace("\'", "")
    song_emb = text_to_emb([song_new])
    similarities = F.cosine_similarity(embeddings_questions, song_emb)
    vals, indices = torch.topk(similarities, k=3)
    results = []
    for i in range(3):
        results.append({
            "label": yes_responses[indices[i]],
            "confidence": round(vals[i].item(), 4)
        })

    return {"status": "success", "matches": results}

