import anthropic
import os
from dotenv import load_dotenv

from create_response import create_anthropic_client
from source_ranking import load_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# System-related parameters
TEMPERATURE = 0.1
SYSTEM_PROMPT_CONFLICT = '''
You are a helpful assistant that determines if two documents contain conflicting factual information.
Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise.
'''

def query_claude_conf(client, model_name, prompt):
    # Query Claude
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT_CONFLICT,
        messages=[
            {'role': 'user', 'content': prompt},
        ]
    )

    return response.content[0].text

def conflict_prompt(higher_text: str, lower_text: str) -> str:
    return (
        "You are a helpful assistant that determines if two documents contain conflicting factual information. "
        "Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise.\n\n"
        f"Higher-priority document:\n{higher_text}\n\n"
        f"Lower-priority document:\n{lower_text}\n\n"
        "Do these two documents conflict? Answer 'Yes' or 'No'."
    )

def has_conflict(client, model_name, higher_text: str, lower_text: str) -> bool:
    prompt = conflict_prompt(higher_text, lower_text)
    response = query_claude_conf(client, model_name, prompt, SYSTEM_PROMPT_CONFLICT)
    answer = response.strip().lower()
    return answer.startswith('yes')

def resolve_conflicts_main(jsonobj):
    jsonobj["data"] = jsonobj["data"].sort(key=lambda x:-x["credibility"])
    filtered = []
    client = create_anthropic_client()
    
    for doc in jsonobj["data"]:
        conflict_found = False
        for kept in filtered:
            if has_conflict(client, 'claude-3-7-sonnet-latest', kept['parsed_text'], doc['parsed_text']):
                print(f"Excluding {doc['url']} due to conflict with {kept['url']}")
                conflict_found = True
                break
        if not conflict_found:
            filtered.append(doc)
    jsonobj["data"] = filtered

    return jsonobj

