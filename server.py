from mcp.server import FastMCP
import requests
from bs4 import BeautifulSoup
import anthropic
import os
import json
import datetime
import torch
import concurrent.futures
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from query import QueryProcessor
from transformers import AutoTokenizer, AutoModel
from googlesearch import search


from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Web Search Server")
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
query_processor = QueryProcessor(claude_client=client)

debug_file_path = os.path.join(os.getcwd(), "server_debug.txt")
f = open(debug_file_path, "w")

# debug to file
def debug_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"{timestamp} - {message}\n")
    f.flush()

debug_log(f"Server starting up. Debug log at: {debug_file_path}")
debug_log(f"Current working directory: {os.getcwd()}")

# Function to load tokenizer and model
def load_tokenizer_and_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tokenizer, model
# Load tokenizer and model outside functions for repeated use
tokenizer, model = load_tokenizer_and_model('sentence-transformers/all-MiniLM-L6-v2')

def process_query(query: str, use_context: bool = True, use_claude: bool = True) -> str:
    """
    Process a user query to enhance it with QueryProcessor.

    Args:
        query: The user's search query
        use_context: Whether to use conversation context (default: True)
        use_claude: Whether to use Claude to enhance the query (default: True)

    Returns:
        JSON string with original and enhanced query
    """
    debug_log(f"process_query called with: {query}, use_context: {use_context}, use_claude: {use_claude}")

    try:
        query_analysis = query_processor.process_query(query, use_context, use_claude)
        debug_log(f"Query processing successful. Enhanced query: {query_analysis.get('enhanced_query', query)}")
        return json.dumps(query_analysis)
    except Exception as e:
        debug_log(f"Error in process_query: {str(e)}")
        return json.dumps({
            "query": query,
            "enhanced_query": query,
            "error": str(e)
        })

def save_results_to_file(results, filename="search_results.json"):
    """Save search results to a JSON file."""
    debug_log(f"Saving results to {filename}")
    
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        debug_log(f"Successfully saved results to {filename}")
    except Exception as e:
        debug_log(f"Error saving results to {filename}: {str(e)}")

# Helper function for mean pooling on model output
def mean_pooling(model_output, attention_mask):
    # Extract last hidden state from model
    token_embeddings = model_output.last_hidden_state 
    # Expand attention mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Calculate mean embeddings
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)  # Prevent division by zero
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings

# Function to make embeddings of documents
def embed_documents(documents, tokenizer, model):
    encoded_input = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
        vectors = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return vectors

# Function to process a batch of URLs for credibility scoring
def process_url_batch(urls_batch):
    """Process a batch of URLs to get credibility scores."""
    try:
        urls_text = "\n".join(urls_batch)
        prompt = f"Assign a credibility score (0.00-1.00) to the following URLs based on the credibility of the website, where 1.00 is most credible:\n{urls_text}\n\n"
        prompt += "Return a JSON object with the URL as the key and the score as the value."
        
        system_prompt = "You are a helpful assistant that measures the credibility scores of URLs. Only respond with the scores in JSON format."
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text.strip("`").strip()
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()
        
        credibility_scores = json.loads(response_text)
        debug_log(f"Successfully assigned credibility scores to {len(credibility_scores)} URLs in batch")
        
        return credibility_scores
    except Exception as e:
        debug_log(f"Error in processing URL batch: {str(e)}")
        return {}  

# Function to assign credibility score to URLs
def assign_credibility_score(urls):
    debug_log(f"assign_credibility_score called with {len(urls)} URLs")
    
    if not urls:
        return {}
    
    try:
        if len(urls) <= 5:
            return process_url_batch(urls)
        
        batches = []
        for i in range(0, len(urls), 5):
            batches.append(urls[i:i+5])
        
        debug_log(f"Split {len(urls)} URLs into {len(batches)} batches for parallel processing")
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_tasks = [executor.submit(process_url_batch, batch) for batch in batches]
            
            # Collect results from all batches
            all_credibility_scores = {}
            for task in concurrent.futures.as_completed(batch_tasks):
                batch_scores = task.result()
                all_credibility_scores.update(batch_scores)
        
        debug_log(f"Successfully assigned credibility scores to {len(all_credibility_scores)} URLs across all batches")
        return all_credibility_scores
        
    except Exception as e:
        debug_log(f"Error in assign_credibility_score: {str(e)}")
        return {}  

def has_conflict(higher_text: str, lower_text: str) -> bool:
    """Determine if two texts contain conflicting information."""
    debug_log(f"Checking for conflicts between documents")
    
    max_length = 5000  # Characters per document
    if len(higher_text) > max_length:
        higher_text = higher_text[:max_length] + "..."
    if len(lower_text) > max_length:
        lower_text = lower_text[:max_length] + "..."
    
    try:
        prompt = (
			"You are a helpful assistant that determines if two documents contain conflicting factual information. "
			"Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise.\n\n"
			f"Higher-priority document:\n{higher_text}\n\n"
			f"Lower-priority document:\n{lower_text}\n\n"
			"Do these two documents conflict? Answer 'Yes' or 'No'."
		)

        system_prompt = "You are a helpful assistant that determines if two documents contain conflicting factual information. Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise."
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.content[0].text.strip().lower()
        return answer.startswith('yes')
    except Exception as e:
        debug_log(f"Error in has_conflict: {str(e)}")
        return False  

def resolve_conflicts(docs):
    """Filter out conflicting documents, keeping the most credible ones."""
    debug_log(f"resolve_conflicts called with {len(docs)} documents")
    
    try:
        # Sort by credibility (highest first)
        docs_by_credibility = sorted(docs, key=lambda x: x.get("credibility", 0), reverse=True)
        
        if len(docs_by_credibility) <= 1:
            return docs_by_credibility
        
        filtered = [docs_by_credibility[0]]  
        
        def check_conflicts(doc, kept_docs):
            for kept in kept_docs:
                if has_conflict(kept["content"], doc["content"]):
                    return True 
            return False  
        
        remaining_docs = docs_by_credibility[1:]
        
        # For each document, check if it conflicts with any of the filtered documents
        with concurrent.futures.ThreadPoolExecutor() as executor:
            conflict_tasks = []
            
            for doc in remaining_docs:
                task = executor.submit(check_conflicts, doc, filtered.copy())
                conflict_tasks.append((doc, task))
                
            for doc, task in conflict_tasks:
                if not task.result():  
                    filtered.append(doc)
                else:
                    debug_log(f"Excluding document due to conflict with higher credibility source")
        
        debug_log(f"After conflict resolution: {len(filtered)} documents remain")
        return filtered
    except Exception as e:
        debug_log(f"Error in resolve_conflicts: {str(e)}")
        return docs 

def rank_sources(docs, query):
    """Rank sources based on relevance to the query and their credibility."""
    debug_log(f"rank_sources called with query: {query}, {len(docs)} documents")
    try:        
        doc_texts = [doc["content"] for doc in docs]

        def embed_single_doc(text, tokenizer, model):
            return embed_documents([text], tokenizer, model)
        
        # Use ThreadPoolExecutor to generate embeddings in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embed_fn = partial(embed_single_doc, tokenizer=tokenizer, model=model)
            doc_embeddings_list = list(executor.map(embed_fn, doc_texts))
        
        doc_embeddings = torch.vstack(doc_embeddings_list)
        query_embedding = embed_documents([query], tokenizer, model)
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        urls = [doc.get("url", "") for doc in docs]
        credibility_scores = assign_credibility_score(urls)
        
        for i, doc in enumerate(docs):
            doc["similarity"] = float(similarities[i])
            url = doc.get("url", "")
            doc["credibility"] = credibility_scores.get(url, 0.5)  # Default to 0.5 if not found
        
        # Resolve conflicts between documents
        docs_without_conflicts = resolve_conflicts(docs)
        ranked_docs = sorted(docs_without_conflicts, key=lambda x: x.get("similarity", 0), reverse=True)
        
        debug_log(f"Successfully ranked {len(ranked_docs)} documents")
        return ranked_docs
    except Exception as e:
        debug_log(f"Error in rank_sources: {str(e)}")
        return docs 

@mcp.tool()
def google_search(query: str, num_results: int = 5) -> list:
    """
    Perform a web search using googlesearch-python and return results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
    
    Returns:
        List of search results with URLs
    """
    debug_log(f"google_search called with query: {query}, num_results: {num_results}")
    
    try:
        results = []
        
        debug_log(f"Performing search with query: {query}")

        # Setting pause to 2.0 to avoid getting blocked by Google
        search_results = list(search(
            query,
            num=num_results,
            start=1,
            stop=num_results+1,
            lang="en",
            pause=2.0
        ))
        
        debug_log(f"Search completed, found {len(search_results)} results")
        
        for url in search_results:
            results.append({
                "url": url
            })
            debug_log(f"Added result: {url}")
        
        return results
        
    except Exception as e:
        debug_log(f"Error in google_search: {str(e)}")
        return [{"error": f"Search failed: {str(e)}"}]
    
@mcp.tool()
def scrape_webpage(url: str) -> dict:
    """
    Scrape content from a webpage.
    
    Args:
        url: The URL to scrape
    
    Returns:
        Dictionary with title, text_content, and links found on the page
    """
    debug_log(f"scrape_webpage called with URL: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        paragraphs = soup.find_all('p')
        text_content = "\n\n".join([p.get_text() for p in paragraphs])
        
        result = {
            "url": url,
            "title": title,
            "content": text_content
        }
        debug_log(f"Successfully scraped webpage, title: {title}")
        return result
        
    except Exception as e:
        debug_log(f"Error in scrape_webpage: {str(e)}")
        return {"error": f"Failed to scrape webpage: {str(e)}"}

@mcp.tool()
def create_summary(query: str, documents: list) -> dict:
    """
    Create a summary of the documents in relation to the query.
    
    Args:
        query: The user's query
        documents: List of documents with content to summarize
    
    Returns:
        A dictionary with the summary and source information
    """
    debug_log(f"create_summary called with query: {query}, documents: {len(documents)}")
    
    try:
        # Extract the top N documents (limiting to top 5)
        top_n = min(5, len(documents))
        top_documents = documents[:top_n]
        
        doc_texts = []
        similarity_scores = []
        credibility_scores = []
        citation_sources = []
        
        for i, doc in enumerate(top_documents):
            if "content" in doc and doc["content"]:
                # Truncate very long documents to avoid exceeding token limits
                content = doc["content"]
                if len(content) > 10000:
                    content = content[:10000] + "..."
                
                doc_texts.append(content)
                similarity_scores.append(doc.get("similarity", 0.0))
                credibility_scores.append(doc.get("credibility", 0.5))
                
                citation_sources.append({
                    "index": i+1,
                    "url": doc.get('url', 'Unknown URL'),
                    "title": doc.get('title', 'Unknown Title'),
                    "similarity": doc.get('similarity', 0.0),
                    "credibility": doc.get('credibility', 0.5),
                    "snippet": doc.get('content', '')[:200] + "..." if doc.get('content') else ""
                })
        
        prompt = ""
        for i, text in enumerate(doc_texts):
            prompt += f"Document {i+1} (sim score = {similarity_scores[i]:.2f}, credibility score = {credibility_scores[i]:.2f}):\n{text}\n\n"
        
        prompt += f"Query: {query}\n\n"
        prompt += "Answer the query by summarizing information from the documents above."
        
        debug_log("Created prompt for summarization")
        
        system_prompt = """
        You are a helpful assistant that summarizes multiple documents in order to answer a user's query. The similarity scores
        and credibility scores for each document are also provided. Answer the query to the best of your understanding. Do not 
        include external information or make up facts. Only include the answer. Do not start with introductions like "based on the provided documents".
        """
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1500,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = response.content[0].text.strip()
        debug_log(f"Created summary of length: {len(summary)}")
        
        return {
            "summary_text": summary,
            "citations": citation_sources
        }
        
    except Exception as e:
        debug_log(f"Error in create_summary: {str(e)}")
        return {
            "summary_text": f"Failed to create summary: {str(e)}",
            "citations": []
        }

def extract_main_content_links(url, soup):
    """
    Extract links only from the main content area of a webpage.
    
    Args:
        url: The URL of the page
        soup: BeautifulSoup object of the page
    
    Returns:
        List of dictionaries with url, anchor_text, and surrounding_context
    """
    debug_log(f"Extracting main content links from {url}")
    
    links = []
    
    try:
        # Try to identify main content area using common patterns
        main_content = None
        
        # Method 1: Look for semantic HTML5 elements
        if not main_content:
            for tag in ['article', 'main', 'section']:
                content_candidates = soup.find_all(tag)
                if content_candidates:
                    # Choose the largest content area by text length
                    main_content = max(content_candidates, key=lambda x: len(x.get_text()))
                    debug_log(f"Found main content using {tag} tag")
                    break
        
        # Method 2: Look for common content class names
        if not main_content:
            for class_name in ['content', 'article', 'post', 'entry', 'story', 'text', 'body']:
                content_candidates = soup.find_all(class_=lambda c: c and class_name in c.lower())
                if content_candidates:
                    main_content = max(content_candidates, key=lambda x: len(x.get_text()))
                    debug_log(f"Found main content using class containing '{class_name}'")
                    break
        
        # Method 3: If still no main content, use heuristics - find div with most text
        if not main_content:
            divs = soup.find_all('div')
            if divs:
                # Filter divs to those with substantial text
                text_divs = [div for div in divs if len(div.get_text()) > 200]
                if text_divs:
                    main_content = max(text_divs, key=lambda x: len(x.get_text()))
                    debug_log("Found main content using text length heuristic")
        
        # If still no main content, fall back to the whole body
        if not main_content:
            main_content = soup.body
            debug_log("Using entire body as main content (fallback)")
        
        # Extract links from the main content
        if main_content:
            for link in main_content.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    href = base_url + href
                
                # Filter out non-http links, anchors, etc.
                if href.startswith(('http://', 'https://')) and '#' not in href:
                    # Get anchor text
                    anchor_text = link.get_text().strip()
                    
                    # Skip links with very short anchor text
                    if len(anchor_text) < 3:
                        continue
                    
                    # Skip common non-content links
                    skip_keywords = ['login', 'sign up', 'subscribe', 'register', 'comment', 
                                    'share', 'privacy', 'terms', 'advertise']
                    if any(keyword in anchor_text.lower() for keyword in skip_keywords):
                        continue
                    
                    # Get surrounding context (up to 100 characters before and after)
                    parent_text = link.parent.get_text()
                    link_position = parent_text.find(anchor_text)
                    start_pos = max(0, link_position - 100)
                    end_pos = min(len(parent_text), link_position + len(anchor_text) + 100)
                    context = parent_text[start_pos:end_pos].strip()
                    
                    links.append({
                        "url": href,
                        "anchor_text": anchor_text,
                        "context": context
                    })
        
        debug_log(f"Extracted {len(links)} links from main content")
        return links
        
    except Exception as e:
        debug_log(f"Error extracting main content links: {str(e)}")
        return []


def prioritize_links_with_claude(query, candidate_links, current_knowledge="", max_links=5):
    """
    Use Claude to prioritize which links to explore next.
    
    Args:
        query: The original search query
        candidate_links: List of potential links with anchor text and context
        current_knowledge: Summary of what we've learned so far
        max_links: Maximum number of links to select (default: 3)
    
    Returns:
        List of selected links with URLs and reasons for selection
    """
    debug_log(f"Prioritizing {len(candidate_links)} links with Claude")
    
    if not candidate_links:
        return []
    
    try:
        # Prepare the prompt
        links_text = ""
        for i, link in enumerate(candidate_links):
            links_text += f"{i+1}. \"{link['anchor_text']}\" - {link['url']}\n"
            links_text += f"   Context: {link['context'][:150]}...\n\n"
        
        # Construct the prompt
        prompt = f"""
        Your task is to select the {max_links} most relevant links to explore next.
        
        ORIGINAL QUERY: {query}
        
        CURRENT KNOWLEDGE: {current_knowledge if current_knowledge else "No information gathered yet."}
        
        CANDIDATE LINKS:
        {links_text}
        
        Select up to {max_links} links that are most likely to contain relevant information about the query.
        If fewer than {max_links} links seem relevant, only select those that are truly relevant.
        For each selected link, provide a brief explanation of why it's relevant.
        
        FORMAT YOUR RESPONSE AS JSON:
        {{
          "selected_links": [
            {{
              "index": 1,
              "reason": "This link likely contains information about X aspect of the query"
            }},
            ...
          ]
        }}
        """
        
        # Send the request to Claude
        system_prompt = "You are a helpful assistant that evaluates the relevance of links for web research. You only respond with valid JSON."
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.1,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract the JSON
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text
        
        # Parse the JSON
        result = json.loads(json_text)
        
        # Map the selected indices back to the original links
        selected_links = []
        for selection in result.get("selected_links", []):
            index = selection.get("index")
            if 1 <= index <= len(candidate_links):
                link_data = candidate_links[index-1].copy()
                link_data["reason"] = selection.get("reason", "No reason provided")
                selected_links.append(link_data)
        
        debug_log(f"Claude selected {len(selected_links)} links to explore next")
        return selected_links
        
    except Exception as e:
        debug_log(f"Error prioritizing links with Claude: {str(e)}")
        return candidate_links[:min(len(candidate_links), max_links)]  # Fallback to first few links

@mcp.tool()
def complete_search_pipeline(query: str, num_results: int = 5, max_depth: int = 5, 
                             max_links_per_page: int = 5, max_parallel_requests: int = 3,
                             follow_links: bool = True) -> dict:
    """
    Run the complete search pipeline with intelligent link exploration and parallel scraping.
    
    Args:
        query: The search query
        num_results: Number of search results to use (default: 5)
        max_depth: Maximum exploration depth (default: 5)
        max_links_per_page: Maximum links to follow from each page (default: 5)
        max_parallel_requests: Maximum number of parallel scraping requests (default: 3)
        follow_links: Whether to follow links from pages or stop at search results (default: True)
    
    Returns:
        Dictionary with all search results and summary
    """
    debug_log(f"complete_search_pipeline called with query: {query}, num_results: {num_results}, max_depth: {max_depth}, follow_links: {follow_links}")
    
    try:
        # Step 1: Perform Google search with the query
        debug_log("Step 1: Performing Google search")
        search_results = google_search(query, num_results)
        
        if not search_results or "error" in search_results[0]:
            error_message = search_results[0].get("error", "Unknown error in search") if search_results else "No search results"
            debug_log(f"Error in search: {error_message}")
            return {
                "error": error_message,
                "search_results": [],
                "content": [],
                "ranked_content": [],
                "summary_text": f"Search failed: {error_message}",
                "citations": []
            }
        
        # Initialize tracking variables
        visited_urls = set()
        content_results = []
        exploration_path = []
        current_knowledge = ""
        
        # Initialize with search results
        urls_to_scrape = []
        for result in search_results:
            url = result.get("url")
            if url and url not in visited_urls:
                urls_to_scrape.append({
                    "url": url, 
                    "depth": 0,
                    "reason": "Initial search result"
                })
                visited_urls.add(url)
        
        # If not following links, set max_depth to 0 to only process search results
        if not follow_links:
            debug_log("Link following disabled, only processing search results")
            max_depth = 0
        
        # Process each depth level
        current_depth = 0
        while current_depth <= max_depth and urls_to_scrape:
            debug_log(f"Processing depth level {current_depth}")
            
            # Filter URLs for the current depth
            current_level_urls = [item for item in urls_to_scrape if item["depth"] == current_depth]
            urls_to_scrape = [item for item in urls_to_scrape if item["depth"] != current_depth]
            
            if not current_level_urls:
                debug_log(f"No URLs to process at depth {current_depth}")
                current_depth += 1
                continue
            
            debug_log(f"Processing {len(current_level_urls)} URLs at depth {current_depth}")
            
            # Process URLs in parallel batches to avoid overwhelming servers
            next_level_candidates = []
            
            # Create batches of URLs to process in parallel
            for i in range(0, len(current_level_urls), max_parallel_requests):
                batch = current_level_urls[i:i+max_parallel_requests]
                debug_log(f"Processing batch of {len(batch)} URLs")
                
                # Scrape URLs in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Map URLs to scraping tasks
                    future_to_url = {
                        executor.submit(scrape_webpage, item["url"]): item 
                        for item in batch
                    }
                    
                    # Process completed tasks
                    for future in concurrent.futures.as_completed(future_to_url):
                        item = future_to_url[future]
                        url = item["url"]
                        depth = item["depth"]
                        reason = item["reason"]
                        
                        try:
                            scrape_result = future.result()
                            
                            if "error" in scrape_result:
                                debug_log(f"Error scraping {url}: {scrape_result.get('error')}")
                                exploration_path.append({
                                    "url": url,
                                    "depth": depth,
                                    "reason": reason,
                                    "success": False,
                                    "error": scrape_result.get('error')
                                })
                                continue
                            
                            if scrape_result.get('content', '') == '':
                                debug_log(f"No content extracted from {url}")
                                continue
                            
                            # Add content to results
                            content_results.append(scrape_result)
                            debug_log(f"Added content from {url}, title: {scrape_result.get('title')}")
                            
                            # Record the successful exploration
                            exploration_path.append({
                                "url": url,
                                "depth": depth,
                                "reason": reason,
                                "success": True,
                                "title": scrape_result.get('title', 'No title')
                            })
                            
                            # Extract links from the page for the next depth level (if following links)
                            if follow_links and depth < max_depth:
                                try:
                                    headers = {
                                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                                    }
                                    response = requests.get(url, headers=headers, timeout=10)
                                    soup = BeautifulSoup(response.text, 'html.parser')
                                    
                                    # Extract links from main content only
                                    content_links = extract_main_content_links(url, soup)
                                    debug_log(f"Extracted {len(content_links)} links from main content of {url}")
                                    
                                    if content_links:
                                        next_level_candidates.append({
                                            "url": url,
                                            "links": content_links
                                        })
                                except Exception as e:
                                    debug_log(f"Error extracting links from {url}: {str(e)}")
                        
                        except Exception as e:
                            debug_log(f"Error processing result for {url}: {str(e)}")
                
                # Update current knowledge summary after each batch
                if len(content_results) >= 2:
                    try:
                        # Create a brief summary of what we know so far
                        knowledge_prompt = f"Briefly summarize what we know about '{query}' based on the following content. Be concise (50-100 words).\n\n"
                        
                        # Include at most 3 content items to avoid token limits
                        for i, content in enumerate(content_results[-3:]):
                            knowledge_prompt += f"Source {i+1}: {content.get('title', 'Untitled')}\n"
                            content_text = content.get('content', '')
                            knowledge_prompt += f"{content_text[:500]}...\n\n"
                        
                        knowledge_response = client.messages.create(
                            model="claude-3-5-sonnet-20240620",
                            max_tokens=200,
                            temperature=0.1,
                            system="You are a helpful assistant that creates concise summaries.",
                            messages=[
                                {"role": "user", "content": knowledge_prompt}
                            ]
                        )
                        
                        current_knowledge = knowledge_response.content[0].text.strip()
                        debug_log(f"Updated current knowledge summary: {current_knowledge}")
                    except Exception as e:
                        debug_log(f"Error updating knowledge summary: {str(e)}")
            
            # Have Claude prioritize the next level links (if following links)
            if follow_links and next_level_candidates:
                debug_log(f"Prioritizing links from {len(next_level_candidates)} pages for depth {current_depth + 1}")
                
                all_candidate_links = []
                for candidate in next_level_candidates:
                    all_candidate_links.extend(candidate["links"])
                
                # If we have too many links, select a diverse sample
                if len(all_candidate_links) > 30:
                    debug_log(f"Too many candidate links ({len(all_candidate_links)}), selecting a sample")
                    # Take every Nth link to get a diverse sample of about 30 links
                    step = max(1, len(all_candidate_links) // 30)
                    sampled_links = [all_candidate_links[i] for i in range(0, len(all_candidate_links), step)][:30]
                    debug_log(f"Selected {len(sampled_links)} sample links")
                    all_candidate_links = sampled_links
                
                # Have Claude select the most relevant links
                if all_candidate_links:
                    selected_links = prioritize_links_with_claude(
                        query, 
                        all_candidate_links, 
                        current_knowledge, 
                        max_links=max_links_per_page
                    )
                    
                    debug_log(f"Claude selected {len(selected_links)} links to explore at depth {current_depth + 1}")
                    
                    # Add selected links to the next depth
                    for link in selected_links:
                        if link["url"] not in visited_urls:
                            urls_to_scrape.append({
                                "url": link["url"],
                                "depth": current_depth + 1,
                                "reason": link.get("reason", "Selected by relevance")
                            })
                            visited_urls.add(link["url"])
            
            # Move to the next depth level
            current_depth += 1
        
        debug_log(f"Navigation complete. Explored {len(visited_urls)} URLs to depth {current_depth - 1}")
        
        if not content_results:
            debug_log("No content could be extracted from any of the pages")
            return {
                "search_results": search_results,
                "exploration_path": exploration_path,
                "follow_links_enabled": follow_links,
                "content": [],
                "ranked_content": [],
                "summary_text": "No content could be extracted from the search results.",
                "citations": []
            }
        
        # Rank sources based on relevance to the query and credibility
        debug_log("Ranking sources by similarity and credibility")
        ranked_content = rank_sources(content_results, query)
        
        # Create summary from ranked content
        debug_log("Creating summary")
        summary_result = create_summary(query, ranked_content[:min(len(ranked_content), 5)])  # Use top 5 sources for summary
        
        result = {
            "search_results": search_results,
            "visited_urls": list(visited_urls),
            "exploration_path": exploration_path,
            "depth_reached": current_depth - 1,
            "follow_links_enabled": follow_links,
            "content": content_results,
            "ranked_content": ranked_content,
            "summary_text": summary_result.get("summary_text", "No summary available"),
            "citations": summary_result.get("citations", [])
        }
        
        save_results_to_file(result)
        
        return result
        
    except Exception as e:
        debug_log(f"Error in complete_search_pipeline: {str(e)}")
        return {
            "error": f"Pipeline failed: {str(e)}",
            "search_results": [],
            "content": [],
            "ranked_content": [],
            "summary_text": f"Search pipeline failed: {str(e)}",
            "citations": []
        }

@mcp.tool()
def get_chat_response(query: str, chat_history: list = None, follow_links: bool = True) -> dict:
    """
    Get a response for a user query with chat history context.
    Uses QueryProcessor to enhance queries before searching.
    
    Args:
        query: The user's query
        chat_history: List of previous queries and responses (optional)
        follow_links: Whether to follow links from pages or stop at search results (default: True)
    
    Returns:
        Dictionary with search results and response
    """
    debug_log(f"get_chat_response called with query: {query}, history length: {len(chat_history) if chat_history else 0}, follow_links: {follow_links}")
    
    try:
        debug_log(f"Original query: {query}")
        use_claude_for_query = True  
        query_analysis_json = process_query(query, use_context=True, use_claude=use_claude_for_query)
        query_analysis = json.loads(query_analysis_json)
        enhanced_query = query
        
        if "error" not in query_analysis:
            enhanced_query = query_analysis.get("enhanced_query", query)
            debug_log(f"Using enhanced query: {enhanced_query}")
        else:
            debug_log(f"Using original query due to error: {query_analysis.get('error')}")

        search_results = complete_search_pipeline(
            query=enhanced_query,
            follow_links=follow_links
        )
        
        search_results["original_query"] = query
        search_results["enhanced_query"] = enhanced_query
        search_results["follow_links_enabled"] = follow_links
        
        if "error" in search_results:
            debug_log(f"Search error: {search_results['error']}")
            return {
                "query": query,
                "enhanced_query": enhanced_query,
                "follow_links_enabled": follow_links,
                "response": f"I encountered an error: {search_results['error']}",
                "search_results": search_results,
                "citations": []
            }
        
        # If we have chat history, include it in the context
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "Previous conversation:\n"
            for i, exchange in enumerate(chat_history):
                history_context += f"User: {exchange.get('query', '')}\n"
                history_context += f"Assistant: {exchange.get('response', '')}\n\n"
        
        summary_text = search_results.get("summary_text", "No summary available")
        citations = search_results.get("citations", [])
        
        response = {
            "query": query,
            "enhanced_query": enhanced_query,
            "follow_links_enabled": follow_links,
            "response": summary_text,
            "search_results": search_results,
            "citations": citations
        }
        
        debug_log(f"Returning chat response with {len(search_results.get('ranked_content', []))} ranked sources")
        return response
        
    except Exception as e:
        debug_log(f"Error in get_chat_response: {str(e)}")
        return {
            "query": query,
            "follow_links_enabled": follow_links,
            "response": f"Sorry, I encountered an error: {str(e)}",
            "search_results": {},
            "citations": []
        }

if __name__ == "__main__":
    debug_log("Starting server.py")
    mcp.run(transport='stdio')