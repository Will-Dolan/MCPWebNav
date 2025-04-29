import anthropic
import os
from dotenv import load_dotenv

from source_ranking import source_ranking_main


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# System-related parameters
TEMPERATURE = 0.1
SYSTEM_PROMPT = '''
You are a helpful assistant that summarizes multiple documents in order to answer a user's query. Answer the query to the
best of your understanding. Do not include external information or make up facts. Only include the answer.
'''


# Helper function to extract top N documents from the ranked data
def extract_top_n_documents(json_ranked_data, top_n):
    # Extract the ranked documents
    ranked_doc_indices = json_ranked_data['ranked_doc_indices']

    # Build up the documents list
    documents = []

    # Iterate through ranked documents (will work even if top_n > len(ranked_doc_indices))
    for index in ranked_doc_indices:
        # Extract the document's text
        text = json_ranked_data['data'][index]['parsed_text']

        # Add to document list
        documents.append(text)

        # Check for top n condition
        if len(documents) == top_n:
            break
    
    # Return the documents
    return documents


# Helper function to create the prompt for Claude
def create_claude_prompt(query, top_n_documents):
    # Create the prompt
    prompt = ''
    for i, doc in enumerate(top_n_documents):
        prompt += f'Document {i+1}:\n{doc}\n\n'
    prompt += f'Query: {query}\n\n'
    prompt += 'Answer the query by summarizing information from the documents above.'

    return prompt


# Create Anthropic client after reading API key from .env file
def create_anthropic_client():
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')

    # Create and return the Anthropic client
    return anthropic.Anthropic(api_key=api_key)


# Helper function to query claude and return the response
def query_claude(client, model_name, prompt):
    # Query Claude
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[
            {'role': 'user', 'content': prompt},
        ]
    )

    return response.content[0].text


def create_response_main():
    # Run source ranking to get the ranked documents
    data_file_path = 'mock_data/source_ranking_input.json'
    json_ranked_data = source_ranking_main(data_file_path=data_file_path, print_results=False)

    # Extract top 5 documents from the data
    top_n = 5
    top_n_documents = extract_top_n_documents(json_ranked_data, top_n)

    # Create the prompt for Claude given the query and top N documents
    query = json_ranked_data['query']
    prompt = create_claude_prompt(query, top_n_documents)

    print(f'Prompt:\n{prompt}\n')


    # Create anthropic client
    client = create_anthropic_client()

    # Anthropic model to use
    model_name = 'claude-3-7-sonnet-latest'

    # Query Claude and get the response
    response = query_claude(client, model_name, prompt)

    print(f'\nResponse:\n{response}')




if __name__ == "__main__":
    create_response_main()