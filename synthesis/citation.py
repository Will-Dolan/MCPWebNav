import anthropic
import os
from dotenv import load_dotenv

from create_response import create_anthropic_client
from source_ranking import load_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# System-related parameters
TEMPERATURE = 0.1

SYSTEM_PROMPT_CITATION = '''
    You are a helpful assistant that generates bibliographic citations.
    Given a URL and an optional style (e.g., APA, MLA, Chicago), produce a formatted citation.
'''
def query_claude_cit(client, model_name, prompt):
    # Query Claude
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT_CITATION,
        messages=[
            {'role': 'user', 'content': prompt},
        ]
    )

    return response.content[0].text

def generate_citation(client, model_name, url: str, style: str = "APA") -> str:
    prompt = (
        f"{SYSTEM_PROMPT_CITATION}\n\n"
        f"URL: {url}\n"
        f"Style: {style}\n\n"
        "Provide the complete citation."
    )
    response = query_claude_cit(client, model_name, prompt)
    return response.completion.strip()

def generate_citation_main(filepath):
  client = create_anthropic_client()
  jsonobj = load_json(filepath)
  for i in range(len(jsonobj["data"])):
    jsonobj["data"][i]["citation"] = generate_citation(client, "claude-3-7-sonnet-latest", jsonobj[i]["url"])
  return jsonobj

