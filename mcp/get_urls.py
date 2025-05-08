import anthropic
import requests
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

def query_common_crawl(url_pattern, crawl_id="CC-MAIN-2025-18"):
	"""Query Common Crawl index with a URL pattern"""
	base_url = f"https://index.commoncrawl.org/{crawl_id}-index"
	params = {
		"url": url_pattern,
		"output": "json"
	}
	
	try:
		response = requests.get(base_url, params=params)
		if response.status_code == 200 and response.text.strip():
			# Common Crawl returns one JSON object per line
			results = [json.loads(line)['url'] for line in response.text.strip().split('\n') if line]
			return results
		else:
			print('failed response: ', response.status_code, response.text)
			return []
	except Exception as e:
		print(f"Exception querying Common Crawl: {e}")
		return []

def query_to_url_patterns(client, query):
	"""Convert a natural language query to relevant URL patterns for Common Crawl"""
	
	system_prompt = """You are a helpful assistant that converts search queries into URL patterns for Common Crawl index searches.
	Follow these guidelines:
	- Generate specific, targeted URL patterns
	- Include domain variations
	- Consider file types and path structures
	- Format output as a JSON list
	"""
		
	prompt = f"""<context>
	<task>Convert search query to Common Crawl URL patterns</task>
	<query>{query}</query>
	<output_format>JSON list of strings</output_format>
	<num_results>5</num_results>
	</context>
		
	Please generate 5 different URL patterns for searching Common Crawl based on this query.
	"""
		
	response = client.messages.create(
		model="claude-3-7-sonnet-20250219",
		max_tokens=1000,
		system=system_prompt,
		messages=[{"role": "user", "content": prompt}]
	)
		
	# Extract JSON list from response
	try:
		content = response.content[0].text
		print(content)
		json_match = re.search(r'\[.*\]', content, re.DOTALL)
		print(json_match)
		if json_match:
			url_patterns = json.loads(json_match.group(0))
			return url_patterns
		else:
			return []
	except Exception as e:
		print(f"Error extracting URL patterns: {e}")
		return []
		
def main():
	client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
		
	while True:
		query = input('search query: ')
		urls = query_to_url_patterns(client, query)
		for url in urls:
			content = query_common_crawl(url)
			print(content[:min(2, len(content))], url)
			print('-------------------------------------------------------')

if __name__=='__main__':
	main()