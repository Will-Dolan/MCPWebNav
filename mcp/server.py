from mcp.server import FastMCP
import requests
import logging
from bs4 import BeautifulSoup
import anthropic
import os
import json
from googlesearch import search 
import datetime
from dotenv import load_dotenv
from mcp.query import QueryProcessor
load_dotenv()

mcp = FastMCP("Web Navigation Server")
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
query_processor = QueryProcessor()

debug_file_path = os.path.join(os.getcwd(), "server_debug.txt")
f = open(debug_file_path, "w")

def debug_log(message):
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	f.write(f"{timestamp} - {message}\n")
	f.flush()

# Log at startup
debug_log(f"Server starting up. Debug log at: {debug_file_path}")
debug_log(f"Current working directory: {os.getcwd()}")

@mcp.tool()
def google_search(query: str, num_results: int = 5) -> list:
    """
    Perform a web search using googlesearch-python and return results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
    
    Returns:
        List of search results with titles and URLs
    """
    debug_log(f"google_search called with query: {query}, num_results: {num_results}")
    
    try:
        results = []
        # Process query through QueryProcessor if it's a natural language query
        # This is purely optional - we'll use the original query if it looks like a search query already
        if ' ' in query and not query.startswith('+') and '+' not in query:
            debug_log("Processing as natural language query")
            query_analysis_json = process_query(query)
            query_analysis = json.loads(query_analysis_json)

            if "error" not in query_analysis:
                search_query = query_analysis.get("search_query", query)
                debug_log(f"Enhanced search query: {search_query}")
                query = search_query

        debug_log(f"Performing search with query: {query}")

        # The googlesearch-python package only returns URLs, not titles
        # Setting pause to 2.0 to avoid getting blocked by Google
        search_results = list(search(
            query,
            num=1,
            start=1,
            stop=5,
            lang="en",
            pause=2.0
        ))
        
        debug_log(f"Search completed, found {len(search_results)} results")
        
        for url in search_results:
            results.append({
                "url": url
            })
            debug_log(f"Added result: {url}")
        
        result_json = json.dumps(results)
        debug_log(f"Returning JSON string: {result_json}")
        return result_json
        
    except Exception as e:
        debug_log(f"Error in google_search: {str(e)}", exc_info=True)
        return json.dumps([{"error": f"Search failed: {str(e)}"}])
    
@mcp.tool()
def scrape_webpage(url: str) -> dict:
    """
    Scrape content from a webpage.
    
    Args:
        url: The URL to scrape
    
    Returns:
        Dictionary with title, text_content, and links found on the page
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract main content (simplified approach)
        # For better content extraction, consider using readability libraries
        paragraphs = soup.find_all('p')
        text_content = "\n\n".join([p.get_text() for p in paragraphs])
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            debug_log('here')
            
            href = link['href']
            debug_log('here')
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                href = base_url + href
            
            # Filter out non-http links, anchors, etc.
            if href.startswith(('http://', 'https://')) and '#' not in href:
                link_text = link.get_text().strip()
                links.append({
                    "url": href,
                    "text": link_text if link_text else "No link text"
                })
        
        return {
            "title": title,
            "text_content": text_content,
            "links": links
        }
        
    except Exception as e:
        return {"error": f"Failed to scrape webpage: {str(e)}"}
    
@mcp.tool()
def analyze_content(query: str, passages: list, links: list) -> dict:
    """
    Analyze content in relation to the user query and decide whether to explore more links
    or extract relevant information.
    
    Args:
        query: The user's original query
        passages: List of text passages from scraped content
        links: List of available links that could be explored
    
    Returns:
        Decision dictionary with action ('explore' or 'extract') and supporting data
    """
    # Format the content for Claude
    context = "\n\n".join(passages)
    links_text = "\n".join([f"{i+1}. {link['text']} - {link['url']}" for i, link in enumerate(links)])
    
    prompt = f"""
    Your task is to analyze content in relation to a user's query and decide whether to:
    1. Explore more links to find better information
    2. Extract relevant information from the current content
    
    USER QUERY: {query}
    
    CONTENT:
    {context}
    
    AVAILABLE LINKS:
    {links_text}
    
    Based on the above, decide if we should:
    - EXPLORE: If the current content doesn't sufficiently answer the query, and there are promising links
    - EXTRACT: If the current content contains information relevant to the query
    
    If EXPLORE, specify which links (by number) should be explored next and why.
    If EXTRACT, specify which parts of the content are most relevant to the query.
    
    DECISION:
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system="You are an AI assistant helping with web research. Your task is to analyze content and decide whether to explore more links or extract relevant information.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        decision_text = response.content[0].text
        
        # Parse the decision
        if "EXPLORE" in decision_text:
            # Extract link numbers if possible
            import re
            link_numbers = re.findall(r'link(?:s)? (?:number )?(\d+)', decision_text.lower())
            links_to_explore = [links[int(num)-1] for num in link_numbers if int(num) <= len(links)]
            
            return {
                "action": "explore",
                "links_to_explore": links_to_explore,
                "reasoning": decision_text
            }
        else:
            return {
                "action": "extract",
                "relevant_content": decision_text,
                "reasoning": decision_text
            }
            
    except Exception as e:
        return {"error": f"Failed to analyze content: {str(e)}"}
    
@mcp.tool()
def process_query(query: str, use_context: bool = True) -> dict:
    """
    Process a user query to enhance it for better search results.

    Args:
        query: The user's search query
        use_context: Whether to use conversation context (default: True)

    Returns:
        Enhanced query analysis with search strategy and refinements
    """
    debug_log(f"process_query called with: {query}, use_context: {use_context}")

    try:
        # Process the query using QueryProcessor
        query_analysis = query_processor.process_query(query, use_context)
        return json.dumps(query_analysis)
    except Exception as e:
        debug_log(f"Error in process_query: {str(e)}")
        return json.dumps({"error": f"Query processing failed: {str(e)}"})

@mcp.tool()
def navigate_web(query: str, max_depth: int = 3) -> dict:
    """
    Run a complete web navigation cycle based on the user query.
    
    Args:
        query: The user's search query
        max_depth: Maximum exploration depth (default: 3)
    
    Returns:
        Results of the web navigation including relevant content
    """
    # Process query first to enhance it
    query_analysis_json = process_query(query)
    query_analysis = json.loads(query_analysis_json)

    # Use the enhanced query if available, otherwise use original
    search_query = query_analysis.get("search_query", query) if "error" not in query_analysis else query
    debug_log(f"Using search query: {search_query}")

    # Start with Google search
    search_results = google_search(search_query)
    debug_log(search_results)
    debug_log(type(search_results))
    
    if not search_results or "error" in search_results[0]:
        return {"error": "Failed to get initial search results"}
    
    # Initialize tracking variables
    depth = 0
    visited_urls = set()
    collected_passages = []
    current_links = search_results
    
    # Main navigation loop
    while depth < max_depth:
        # If we have no more links to explore, end the cycle
        if not current_links:
            break
        
        # Choose the next link to explore (in a more sophisticated implementation,
        # you might use Claude to prioritize which link to explore next)
        debug_log(current_links[0])
        next_link = current_links[0]["url"]
        current_links = current_links[1:]  # Remove the link we're about to explore
        
        # Skip if we've already visited this URL
        if next_link in visited_urls:
            continue
            
        visited_urls.add(next_link)
        
        # Scrape the webpage
        debug_log(next_link)
        scrape_result = scrape_webpage(next_link)
        
        if "error" in scrape_result:
            continue
        
        # Add content to collected passages
        collected_passages.append(scrape_result["text_content"])
        
        # Analyze content and decide next steps
        analysis = analyze_content(
            query=query,
            passages=collected_passages,
            links=scrape_result["links"]
        )
        
        if "error" in analysis:
            continue
        
        # If Claude decides to extract, return the results
        if analysis["action"] == "extract":
            return {
                "action": "extract",
                "query": query,
                "depth_reached": depth,
                "visited_urls": list(visited_urls),
                "extracted_content": analysis["relevant_content"]
            }
        
        # Otherwise, add new links to explore
        elif analysis["action"] == "explore":
            current_links.extend(analysis["links_to_explore"])
        
        # Increment depth
        depth += 1
    
    # If we've reached max depth or run out of links, return what we have
    return {
        "action": "max_depth_reached",
        "query": query,
        "depth_reached": depth,
        "visited_urls": list(visited_urls),
        "collected_passages": collected_passages
    }

if __name__ == "__main__":
    mcp.run(transport='stdio')