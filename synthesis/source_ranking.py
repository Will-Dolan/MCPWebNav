import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import anthropic

TEMPERATURE = 0.1
SYSTEM_PROMPT = '''
You are a helpful assistant that measures the credibility scores of URLs. Only respond with the scores in JSON format.
'''

# Load in the JSON file and return the data
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Load in tokenizer and model
def load_tokenizer_and_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tokenizer, model


# Function to assign a credibility score to a list of URLs
def assign_credibility_score(urls):
    # Prompt for Claude
    prompt = f"Assign a credibility score (0.00-1.00) to the following URLs based on the credibility of the website, where 1.00 is most credible:\n{urls}\n\n"
    prompt += "Return a JSON object with the URL as the key and the score as the value."

     # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')

    # Create and return the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Query Claude
    response = client.messages.create(
        model='claude-3-7-sonnet-latest',
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[
            {'role': 'user', 'content': prompt},
        ]
    )

    response = response.content[0].text

    response = response.strip("`").strip()
    if response.startswith("json"):
        response = response[4:].strip()


    # Extract the credibility scores from the response
    try:
        credibility_scores = json.loads(response)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response}")
        credibility_scores = {}
    # Print the credibility scores
    # print(f'Credibility Scores:\n{credibility_scores}')

    return credibility_scores


# Helper function to perform mean pooling on the model output
def mean_pooling(model_output, attention_mask):
    # Extract last hidden state form model, (batch_size, seq_len, hidden_size)
    token_embeddings = model_output.last_hidden_state 
    # Extract attention mask and expand it to match the token embeddings shape
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Sum up the token embeddings and divide by the sum of the attention mask to get mean embeddings
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)  # Prevent division by zero
    mean_embeddings = sum_embeddings / sum_mask

    # Return the mean embeddings
    return mean_embeddings
        

# Function to make embeddings of documents given a list of text
def embed_documents(documents, tokenizer, model):
    # Tokenize the documents
    encoded_input = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Disable gradient calculations since we are not training the model
    with torch.no_grad():
        model_output = model(**encoded_input)
        vectors = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return vectors


# Print out ranking results
def print_ranking_results(json_data):
    print(f'Query: {json_data["query"]}')
    print(f'\nRanking Results:')
    print(f'Ranked Document Indices: {json_data["ranked_doc_indices"]}')
    for i, doc_item in enumerate(json_data['data']):
        print(f"Document {i}:")
        print(f"  URL: {doc_item['url']}")
        # print(f"  Text: {doc_item['parsed_text']}")
        print(f"  Similarity: {doc_item['similarity']}")
        print(f"  Rank: {doc_item['rank']}")
    print()


def source_ranking_main(data_file_path=None, print_results=True):

    # If no data file path provided, use testing one
    if not data_file_path:
        # File path for testing
        data_file_path = 'mock_data/source_ranking_input.json'
        print(f'Using testing file path: {data_file_path}')

    # Load the data
    json_data = load_json(data_file_path)

    # Extract query
    query = json_data['query']

    # Data - assume fairly short (within a paragraph)
    # If longer, may need to split into smaller chunks
    # print(json_data)

    # Model: use sentence transformers to embed data (ex: sentence-transformers/all-MiniLM-L6-v2)
    tokenizer, model = load_tokenizer_and_model('sentence-transformers/all-MiniLM-L6-v2')

    # Extract the list of documents 
    documents = [item['parsed_text'] for item in json_data['data']]
    
    # Embed the documents, shape: (num_documents, embedding_dim)
    document_embeddings = torch.vstack([embed_documents(doc, tokenizer, model) for doc in documents])

    # Create embedding for the query, shape: (1, embedding_dim)
    query_embedding = embed_documents([query], tokenizer, model)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    # Based on cosine similarities, rank the documents
    # ranked_doc_indices[0] is the index of the most similar document
    # ranked_doc_indices[-1] is the index of the least similar document
    ranked_doc_indices = cosine_similarities.argsort()[::-1].tolist()

    # Add information to the JSON data
    json_data['ranked_doc_indices'] = ranked_doc_indices

    for i, doc_item in enumerate(json_data['data']):
        doc_item['similarity'] = cosine_similarities[i]
        doc_item['rank'] = ranked_doc_indices.index(i)

    
    # Print out ranking results
    if print_results:
        print_ranking_results(json_data)

    # Assign credibility scores to the URLs
    urls = [item['url'] for item in json_data['data']]
    credibilities_json = assign_credibility_score(urls)

    # Add credibility scores to the JSON data
    for i, doc_item in enumerate(json_data['data']):
        url = doc_item['url']
        if url in credibilities_json:
            doc_item['credibility'] = credibilities_json[url]
        else:
            doc_item['credibility'] = None

    # print(json_data)

    # Return the JSON data
    return json_data


if __name__ == '__main__':
    source_ranking_main()