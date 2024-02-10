import os

from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client import models, QdrantClient
import torch
from dotenv import load_dotenv
from openai import OpenAI

# model_id = "naver/splade-cocondenser-ensembledistil"
model_id = "naver/efficient-splade-VI-BT-large-query"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

load_dotenv()

openAIClient = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url="https://e9d74ef0-b9f2-4b44-b5f0-e22ea1d6fc34.europe-west3-0.gcp.cloud.qdrant.io:6334",
                      api_key=qdrant_api_key,
                      prefer_grpc=True)
# client = QdrantClient("localhost")

COLLECTION_NAME = "hybrid_collection"


def compute_dense_vector(prompt, model="text-embedding-3-large"):
    prompt = prompt.replace("\n", " ")
    return openAIClient.embeddings.create(input=prompt, model=model).data[0].embedding

def compute_sparse_vector(prompt, tokenizer=tokenizer, model=model):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.

    Args:
    logits (torch.Tensor): The logits output from a model.
    attention_mask (torch.Tensor): The attention mask corresponding to the input tokens.

    Returns:
    torch.Tensor: Computed vector.
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    # MLM (Masked Language Model) Output. The model predicts the probability distribution (logits) across the entire vocabulary,
    # typically consisting of 30,522 tokens (the BERT vocabulary size).
    # The highest activation in this distribution corresponds to the prediction for the masked token position.
    output = model(**tokens)

    logits, attention_mask = output.logits, tokens.attention_mask
    # SPLADE takes the probability distribution from the MLM step
    # and aggregates them into a single distribution called the “Importance Estimation.”.
    # This distribution represents the sparse vector, highlighting relevant tokens that may not exist in the original input sequence.
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens


documents = [
    {
        # "name": "The Time Machine",
        "name": "מכונת הזמן",
        "description": "A man travels through time and witnesses the evolution of humanity.",
        "author": "H.G. Wells",
        "year": 1895,
    },
    {
        # "name": "Ender's Game",
        "name": "מחשק מכור",
        "description": "A young boy is trained to become a military leader in a war against an alien race.",
        "author": "Orson Scott Card",
        "year": 1985,
    },
    {
        # "name": "Brave New World",
        "name": "עולם חדש ואמיץ",
        "description": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
        "author": "Aldous Huxley",
        "year": 1932,
    },
    {
        # "name": "The Hitchhiker's Guide to the Galaxy",
        "name" : "מדריך הטרמפיסט לגלקסיה",
        "description": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
        "author": "Douglas Adams",
        "year": 1979,
    },
    {
        #"name": "Dune",
        "name": "חוֹלִית",
        "description": "A desert planet is the site of political intrigue and power struggles.",
        "author": "Frank Herbert",
        "year": 1965,
    },
    {
        # "name": "Foundation",
        "name": "קרן",
        "description": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
        "author": "Isaac Asimov",
        "year": 1951,
    },
    {
        #"name": "Snow Crash",
        "name": "התרסקות שלג",
        "description": "A futuristic world where the internet has evolved into a virtual reality metaverse.",
        "author": "Neal Stephenson",
        "year": 1992,
    },
    {
        #"name": "Neuromancer",
        "name": "נוירומנסר",
        "description": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
        "author": "William Gibson",
        "year": 1984,
    },
    {
        # "name": "The War of the Worlds",
        "name": "מלחמת העולמות",
        "description": "A Martian invasion of Earth throws humanity into chaos.",
        "author": "H.G. Wells",
        "year": 1898,
    },
    {
        # "name": "The Hunger Games",
        "name": "משחקי הרעב",
        "description": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
        "author": "Suzanne Collins",
        "year": 2008,
    },
    {
        # "name": "The Andromeda Strain",
        "name": "זן אנדרומדה",
        "description": "A deadly virus from outer space threatens to wipe out humanity.",
        "author": "Michael Crichton",
        "year": 1969,
    },
    {
        # "name": "The Left Hand of Darkness",
        "name": "יד שמאל של החושך",
        "description": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
        "author": "Ursula K. Le Guin",
        "year": 1969,
    },
    {
        #"name": "The Three-Body Problem",
        "name": "בעיית שלושת הגופים",
        "description": "Humans encounter an alien civilization that lives in a dying system.",
        "author": "Liu Cixin",
        "year": 2008,
    },
]

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "text-dense": models.VectorParams(
            size=3072,  # OpenAI Embeddings
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    }
)

pts = []
for idx, doc in enumerate(documents):
    vec, tokens = compute_sparse_vector(doc["name"], tokenizer=tokenizer, model=model)
    indices = vec.nonzero().numpy().flatten()
    values = vec.detach().numpy()[indices]
    pt = models.PointStruct(
        id=idx,
        vector={
            "text-dense": compute_dense_vector(doc["name"]),
            "text-sparse": models.SparseVector(
                indices=indices,
                values=values
            )
        },
        payload={
            "name": doc["name"],
            "description": doc["description"],
            "author": doc["author"],
            "year": doc["year"],
        }
    )
    pts.append(pt)


client.upsert(
    collection_name=COLLECTION_NAME,
    points=pts,
)


prompt = "הרעב"
query_vec, query_tokens = compute_sparse_vector(prompt)
query_indices = query_vec.nonzero().numpy().flatten()
query_values = query_vec.detach().numpy()[query_indices]

search_queries = [
    models.SearchRequest(
      vector=models.NamedVector(
          name="text-dense",
          vector=compute_dense_vector(prompt)
        ),
      limit=3,
      with_payload=True
    ),
    models.SearchRequest(
        vector=models.NamedSparseVector(
            name="text-sparse",
            vector=models.SparseVector(
                indices=query_indices,
                values=query_values
            )
        ),
        limit=3,
        with_payload=True
    )
]

results = client.search_batch(
    collection_name=COLLECTION_NAME,
    requests=search_queries
)

print(results)