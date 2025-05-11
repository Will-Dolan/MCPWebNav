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
from query import QueryProcessor  # Import the updated QueryProcessor

load_dotenv()

mcp = FastMCP("Web Navigation Server")
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize QueryProcessor with the Claude client
query_processor = QueryProcessor(claude_client=client)

debug_file_path = os.path.join(os.getcwd(), "server_debug.txt")
f = open(debug_file_path, "w")

def debug_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"{timestamp} - {message}\n")
    f.flush()

# Log at startup
debug_log(f"Server starting up. Debug log at: {debug_file_path}")
debug_log(f"Current working directory: {os.getcwd()}")

def save_results_to_file(query, enhanced_query, filename="results.json"):
    """
    Save the query and enhanced query to a JSON file.
    
    Args:
        query: The original user query
        enhanced_query: The query enhanced by Claude
        filename: The filename to save to (default: results.json)
    """
    debug_log(f"Saving results to {filename}")
    
    # Create a simple results dictionary
    results = {
        "query": query,
        "enhanced_query": enhanced_query
    }
    
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        debug_log(f"Successfully saved results to {filename}")
    except Exception as e:
        debug_log(f"Error saving results to {filename}: {str(e)}")

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
                enhanced_query = query_analysis.get("enhanced_query", query)
                debug_log(f"Enhanced query: {enhanced_query}")
                query = enhanced_query

        debug_log(f"Performing search with query: {query}")

        # The googlesearch-python package only returns URLs, not titles
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
        
        result_json = json.dumps(results)
        debug_log(f"Returning JSON string: {result_json}")
        return result_json
        
    except Exception as e:
        debug_log(f"Error in google_search: {str(e)}")
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
            href = link['href']
            
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
def analyze_content(query: str, passages: list, links: list, passage_sources: list = None) -> dict:
    """
    Analyze content in relation to the user query and decide whether to explore more links
    or extract relevant information, or both.
    
    Args:
        query: The user's original query
        passages: List of text passages from scraped content
        links: List of available links that could be explored
        passage_sources: List of source information for each passage (optional)
    
    Returns:
        Decision dictionary with action ('explore', 'extract', or 'both') and supporting data
        including exact paragraphs extracted from the original text with their source URLs
    """
    # Format the content for Claude
    context = "\n\n".join(passages)
    links_text = "\n".join([f"{i+1}. {link['text']} - {link['url']}" for i, link in enumerate(links)])
    
    # Break the content into paragraphs for easier reference
    all_paragraphs = []
    paragraph_sources = []  # Track the source of each paragraph
    
    for i, passage in enumerate(passages):
        # Split each passage into paragraphs (sequences separated by double newlines)
        paragraphs = [p.strip() for p in passage.split("\n\n") if p.strip()]
        all_paragraphs.extend(paragraphs)
        
        # If passage_sources is provided, associate each paragraph with its source
        if passage_sources and i < len(passage_sources):
            paragraph_sources.extend([passage_sources[i]] * len(paragraphs))
        else:
            # Default empty source if not provided
            paragraph_sources.extend([{"url": "unknown", "title": "Unknown Source"}] * len(paragraphs))
    
    # Create a reference index of paragraphs for Claude
    paragraphs_text = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(all_paragraphs)])
    
    prompt = f"""
    Your task is to analyze content in relation to a user's query and decide whether to:
    1. Explore more links to find better information
    2. Extract relevant information from the current content
    3. Both extract relevant information AND explore more links
    
    USER QUERY: {query}
    
    CONTENT:
    {paragraphs_text}
    
    AVAILABLE LINKS:
    {links_text}
    
    Based on the above, decide if we should:
    - EXPLORE: If the current content doesn't sufficiently answer the query, and there are promising links
    - EXTRACT: If the current content contains information relevant to the query
    - BOTH: If the current content contains some relevant information but exploring additional links would provide more complete information
    
    If EXPLORE or BOTH, specify which links (by number) should be explored next and why.
    
    If EXTRACT or BOTH, identify the exact paragraphs that contain relevant information by referring to their paragraph numbers like [0], [1], etc.
    
    Your response must follow this JSON format:
    ```json
    {{
        "decision": "EXPLORE|EXTRACT|BOTH",
        "explanation": "Your reasoning for the decision",
        "links_to_explore": [1, 2, 3],  // Only include if decision is EXPLORE or BOTH, list link numbers to explore
        "relevant_paragraphs": [0, 5, 9],  // Only include if decision is EXTRACT or BOTH, list paragraph numbers
        "summary": "A brief summary of the relevant information found"  // Only include if decision is EXTRACT or BOTH
    }}
    ```
    
    Only respond with valid JSON in the exact format specified above. No other text outside the JSON block.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1500,
            temperature=0,
            system="You are an AI assistant helping with web research. Your task is to analyze content and decide whether to explore more links or extract relevant information or both. You must respond with properly formatted JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        decision_text = response.content[0].text
        
        # Extract the JSON string from the response, removing any markdown code block decorators
        json_str = decision_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        
        json_str = json_str.strip()
        
        # Parse the JSON response
        decision_data = json.loads(json_str)
        
        # Extract relevant paragraphs text if specified, along with their sources
        extracted_paragraphs_with_sources = []
        if "relevant_paragraphs" in decision_data and decision_data["relevant_paragraphs"]:
            for idx in decision_data["relevant_paragraphs"]:
                if idx < len(all_paragraphs):
                    extracted_paragraphs_with_sources.append({
                        "paragraph": all_paragraphs[idx],
                        "source": paragraph_sources[idx] if idx < len(paragraph_sources) else {"url": "unknown", "title": "Unknown Source"}
                    })
        
        # Prepare links to explore if specified
        if "links_to_explore" in decision_data and decision_data["links_to_explore"]:
            links_to_explore = [links[idx-1] for idx in decision_data["links_to_explore"] if 0 < idx <= len(links)]
        else:
            links_to_explore = []
        
        decision = decision_data.get("decision", "").upper()
        
        # Format the result based on the decision
        if decision == "BOTH":
            return {
                "action": "both",
                "links_to_explore": links_to_explore,
                "extracted_paragraphs": extracted_paragraphs_with_sources,
                "paragraph_indices": decision_data.get("relevant_paragraphs", []),
                "summary": decision_data.get("summary", ""),
                "reasoning": decision_data.get("explanation", "")
            }
        elif decision == "EXPLORE":
            return {
                "action": "explore",
                "links_to_explore": links_to_explore,
                "reasoning": decision_data.get("explanation", "")
            }
        else:  # EXTRACT
            return {
                "action": "extract",
                "extracted_paragraphs": extracted_paragraphs_with_sources,
                "paragraph_indices": decision_data.get("relevant_paragraphs", []),
                "summary": decision_data.get("summary", ""),
                "reasoning": decision_data.get("explanation", "")
            }
            
    except Exception as e:
        debug_log(f"Error in analyze_content: {str(e)}")
        return {"error": f"Failed to analyze content: {str(e)}"}
    
@mcp.tool()
def process_query(query: str, use_context: bool = True, use_claude: bool = True) -> dict:
    """
    Process a user query to enhance it with Claude.

    Args:
        query: The user's search query
        use_context: Whether to use conversation context (default: True)
        use_claude: Whether to use Claude to enhance the query (default: True)

    Returns:
        Dictionary with original and enhanced query
    """
    debug_log(f"process_query called with: {query}, use_claude: {use_claude}")

    try:
        # Process the query using QueryProcessor with Claude enhancement
        query_analysis = query_processor.process_query(query, use_context, use_claude)
        debug_log(f"Query processing successful. Enhanced query: {query_analysis.get('enhanced_query', query)}")
        return json.dumps(query_analysis)
    except Exception as e:
        debug_log(f"Error in process_query: {str(e)}")
        return json.dumps({"error": f"Query processing failed: {str(e)}"})

@mcp.tool()
def navigate_web(query: str, max_depth: int = 3, use_claude_for_query: bool = True) -> dict:
    """
    Run a complete web navigation cycle based on the user query.
    
    Args:
        query: The user's search query
        max_depth: Maximum exploration depth (default: 3)
        use_claude_for_query: Whether to use Claude to enhance the query (default: True)
    
    Returns:
        Results of the web navigation including relevant content with source URLs
    """
    debug_log(f"navigate_web called with: {query}, max_depth: {max_depth}, use_claude_for_query: {use_claude_for_query}")
    
    # Process query first to enhance it
    debug_log(f"Original query: {query}")
    query_analysis_json = process_query(query, use_context=True, use_claude=use_claude_for_query)
    query_analysis = json.loads(query_analysis_json)

    # Use the enhanced query if available, otherwise use original
    # Initialize variables to store query information
    enhanced_query = query
    
    if "error" not in query_analysis:
        # Get enhanced query
        enhanced_query = query_analysis.get("enhanced_query", query)
        debug_log(f"Using enhanced query: {enhanced_query}")
    else:
        debug_log(f"Using original query due to error: {query_analysis.get('error')}")

    # Start with Google search - use the enhanced query
    search_results = google_search(enhanced_query)
    debug_log(f"Search results: {search_results}")
    
    try:
        search_results_parsed = json.loads(search_results)
        if not search_results_parsed or "error" in search_results_parsed[0]:
            error_result = {
                "error": "Failed to get initial search results",
                "query": query,
                "enhanced_query": enhanced_query
            }
            save_results_to_file(query, enhanced_query)
            return error_result
    except Exception as e:
        debug_log(f"Error parsing search results: {str(e)}")
        error_result = {
            "error": f"Failed to parse search results: {str(e)}",
            "query": query,
            "enhanced_query": enhanced_query
        }
        save_results_to_file(query, enhanced_query)
        return error_result
    
    # Initialize tracking variables
    depth = 0
    visited_urls = set()
    collected_passages = []  # Will store dicts with URL and content
    collected_extractions = []  # For storing extracted relevant content
    current_links = search_results_parsed
    
    # Main navigation loop
    while depth < max_depth:
        # If we have no more links to explore, end the cycle
        if not current_links:
            debug_log("No more links to explore")
            break
        
        # Choose the next link to explore
        debug_log(f"Current depth: {depth}")
        next_link = current_links[0]["url"]
        debug_log(f"Next link to explore: {next_link}")
        current_links = current_links[1:]  # Remove the link we're about to explore
        debug_log(f"Remaining links: {len(current_links)}")
        
        # Skip if we've already visited this URL
        if next_link in visited_urls:
            debug_log(f"Skipping already visited URL: {next_link}")
            continue
            
        visited_urls.add(next_link)
        
        # Scrape the webpage
        debug_log(f"Scraping webpage: {next_link}")
        scrape_result = scrape_webpage(next_link)
        
        if "error" in scrape_result:
            debug_log(f"Error scraping webpage: {scrape_result['error']}")
            continue
        
        # Add content to collected passages with URL information
        current_passage = {
            "url": next_link,
            "title": scrape_result["title"],
            "content": scrape_result["text_content"]
        }
        collected_passages.append(current_passage)
        debug_log(f"Added passage from {next_link}, title: {scrape_result['title']}")
        
        # For analysis, we'll use all scraped content
        # Extract just the text content for analysis
        analysis_passages = [p["content"] for p in collected_passages]
        
        # Also pass the source information for each passage
        passage_sources = [{"url": p["url"], "title": p["title"]} for p in collected_passages]
        
        # Analyze content and decide next steps
        debug_log("Analyzing content")
        analysis = analyze_content(
            query=query,  # Use original query for relevance judgment
            passages=analysis_passages,
            links=scrape_result["links"],
            passage_sources=passage_sources
        )
        
        if "error" in analysis:
            debug_log(f"Error in content analysis: {analysis['error']}")
            continue
        
        debug_log(f"Analysis result action: {analysis['action']}")
        
        # If Claude decides to extract only, return the results
        if analysis["action"] == "extract":
            debug_log("Returning extraction results")
            final_result = {
                "query": query,
                "enhanced_query": enhanced_query,
                "action": "extract",
                "depth_reached": depth,
                "visited_urls": list(visited_urls),
                "current_url": next_link,
                "current_title": scrape_result["title"],
                "extracted_paragraphs": analysis.get("extracted_paragraphs", []),
                "summary": analysis.get("summary", ""),
                "reasoning": analysis.get("reasoning", ""),
                "collected_passages": collected_passages
            }
            
            # Save just query and enhanced query to results.json
            save_results_to_file(query, enhanced_query)
                
            return final_result
        
        # If Claude decides to both extract and explore
        elif analysis["action"] == "both":
            debug_log("Both extracting content and adding links to explore")
            # Add the extracted content to our collection
            collected_extractions.extend(analysis.get("extracted_paragraphs", []))
            
            # Add new links to explore
            if analysis.get("links_to_explore"):
                debug_log(f"Adding {len(analysis['links_to_explore'])} links to explore")
                current_links.extend(analysis["links_to_explore"])
            else:
                debug_log("No additional links to explore")
        
        # Otherwise, just add new links to explore
        elif analysis["action"] == "explore":
            if analysis.get("links_to_explore"):
                debug_log(f"Adding {len(analysis['links_to_explore'])} links to explore")
                current_links.extend(analysis["links_to_explore"])
            else:
                debug_log("No additional links to explore")
        
        # Increment depth
        depth += 1
    
    # If we've reached max depth or run out of links
    debug_log("Reached max depth or no more links")
    debug_log(f"Collected {len(collected_extractions)} extractions")
    debug_log(f"Visited {len(visited_urls)} URLs")
    
    # Save just the query information to results.json
    save_results_to_file(query, enhanced_query)
    
    # Prepare the final result for return
    final_result = {}
    
    # If we collected any extractions, return those
    if collected_extractions:
        debug_log("Returning with collected extractions")
        final_result = {
            "query": query,
            "enhanced_query": enhanced_query,
            "action": "completed_with_extractions",
            "depth_reached": depth,
            "visited_urls": list(visited_urls),
            "extracted_content": collected_extractions,
            "collected_passages": collected_passages
        }
    # Otherwise return what we have
    else:
        debug_log("Returning max depth reached with collected passages")
        final_result = {
            "query": query,
            "enhanced_query": enhanced_query,
            "action": "max_depth_reached",
            "depth_reached": depth,
            "visited_urls": list(visited_urls),
            "collected_passages": collected_passages
        }
    
    return final_result

if __name__ == "__main__":
    mcp.run(transport='stdio')