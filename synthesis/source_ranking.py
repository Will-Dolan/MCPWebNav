import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


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
        print(f"Document {i+1}:")
        print(f"  URL: {doc_item['url']}")
        # print(f"  Text: {doc_item['parsed_text']}")
        print(f"  Similarity: {doc_item['similarity']}")
        print(f"  Rank: {doc_item['rank']}")
    print()


def main():
    # File path for testing
    testing_file_path = 'mock_data/source_ranking_input.json'

    # Load the data
    json_data = load_json(testing_file_path)

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
    print_ranking_results(json_data)


if __name__ == '__main__':
    main()