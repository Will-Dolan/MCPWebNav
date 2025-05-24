"""
Web Search MCP Server
Implements MCP tools for web search, scraping, and content summarization
"""
import os
import json
import requests
import datetime
import torch
import concurrent.futures
from functools import partial
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModel
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the MCP server with a name
mcp = FastMCP("Web Search MCP Server")

# Set up logging
log_file_path = os.path.join(os.getcwd(), "mcp_server_debug.txt")
f = open(log_file_path, "w")

def debug_log(message):
    """Log debug messages to a file with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"{timestamp} - {message}"
    f.write(f"{log_msg}\n")
    f.flush()
debug_log(f"MCP Server starting up. Debug log at: {log_file_path}")
debug_log(f"Current working directory: {os.getcwd()}")

# Initialize Anthropic client if API key is available
client = None
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_api_key:
    client = Anthropic(api_key=anthropic_api_key)
    debug_log("Anthropic client initialized successfully")
else:
    debug_log("Warning: ANTHROPIC_API_KEY not found, Claude-related functions will be unavailable")

# Load tokenizer and model for embedding
def load_tokenizer_and_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the tokenizer and model for text embeddings"""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tokenizer, model

# Initialize tokenizer and model
tokenizer, model = load_tokenizer_and_model()
debug_log("Tokenizer and model loaded successfully")

# Helper function for mean pooling on model output
def mean_pooling(model_output, attention_mask):
    """Create mean pooling for embeddings"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)  # Prevent division by zero
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

# Function to make embeddings of documents
def embed_documents(documents, tokenizer, model):
    """Create embeddings for a list of documents"""
    encoded_input = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
        vectors = mean_pooling(model_output, encoded_input['attention_mask'])
    return vectors

@mcp.tool(description="Search the web for information using search engines")
def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a web search using search engines.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
    
    Returns:
        Dictionary with search results containing URLs
    """
    debug_log(f"=== WEB_SEARCH TOOL CALLED ===")
    debug_log(f"Query: '{query}'")
    debug_log(f"Number of results requested: {num_results}")
    
    try:
        from googlesearch import search
        results = []
        
        debug_log(f"Performing search with query: {query}")

        # Setting pause to avoid getting blocked
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
        
        debug_log(f"web_search completed successfully with {len(results)} results")
        return {"results": results}
        
    except Exception as e:
        debug_log(f"Error in web_search: {str(e)}")
        return {"error": f"Search failed: {str(e)}", "results": []}

@mcp.tool(description="Fetch and extract content from a webpage")
def web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a webpage.
    
    Args:
        url: The URL to scrape
    
    Returns:
        Dictionary with title, text_content from the page
    """
    debug_log(f"=== WEB_FETCH TOOL CALLED ===")
    debug_log(f"URL: {url}")
    
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
        debug_log(f"Error in web_fetch: {str(e)}")
        return {"error": f"Failed to scrape webpage: {str(e)}"}

@mcp.tool(description="Assign credibility scores to a list of URLs using Claude")
def assign_credibility_score(urls: List[str]) -> Dict[str, float]:
    """
    Assign credibility scores to a list of URLs using Claude.
    
    Args:
        urls: List of URLs to score for credibility
    
    Returns:
        Dictionary mapping URLs to credibility scores (0.0 to 1.0)
    """
    debug_log(f"=== CREDIBILITY_SCORE TOOL CALLED ===")
    debug_log(f"Number of URLs to score: {len(urls)}")
    if urls:
        debug_log(f"URLs: {urls[:3]}{'...' if len(urls) > 3 else ''}")
    
    if not urls:
        return {}
    
    if not client:
        debug_log("Cannot assign credibility scores: Anthropic client not initialized")
        return {url: 0.5 for url in urls}  # Default score when client is unavailable
    
    try:
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
        return {url: 0.5 for url in urls}  # Default score on error

@mcp.tool(description="Rank content based on relevance to a query and credibility")
def rank_content(documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Rank content based on relevance to the query and credibility.
    
    Args:
        documents: List of documents with content to rank
        query: The original search query
    
    Returns:
        List of documents ranked by relevance and credibility
    """
    debug_log(f"=== RANK_CONTENT TOOL CALLED ===")
    debug_log(f"Query: '{query}'")
    debug_log(f"Number of documents to rank: {len(documents)}")
    try:
        if not documents:
            return []
            
        doc_texts = [doc.get("content", "") for doc in documents]

        def embed_single_doc(text, tokenizer, model):
            return embed_documents([text], tokenizer, model)
        
        # Use ThreadPoolExecutor to generate embeddings in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embed_fn = partial(embed_single_doc, tokenizer=tokenizer, model=model)
            doc_embeddings_list = list(executor.map(embed_fn, doc_texts))
        
        doc_embeddings = torch.vstack(doc_embeddings_list)
        query_embedding = embed_documents([query], tokenizer, model)
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        urls = [doc.get("url", "") for doc in documents]
        credibility_scores = {}
        
        if client:
            credibility_scores = assign_credibility_score(urls)
        else:
            credibility_scores = {url: 0.5 for url in urls}  # Default score when client is unavailable
        
        for i, doc in enumerate(documents):
            doc["similarity"] = float(similarities[i])
            url = doc.get("url", "")
            doc["credibility"] = credibility_scores.get(url, 0.5)  # Default to 0.5 if not found
        
        # Resolve conflicts between documents (if implemented)
        # For now, just sort by similarity
        ranked_docs = sorted(documents, key=lambda x: x.get("similarity", 0), reverse=True)
        
        debug_log(f"Successfully ranked {len(ranked_docs)} documents")
        return ranked_docs
    except Exception as e:
        debug_log(f"Error in rank_content: {str(e)}")
        return documents  # Return original documents on error

@mcp.tool(description="Create a summary of documents in relation to a query")
def create_summary(query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of the documents in relation to the query.
    
    Args:
        query: The user's query
        documents: List of documents with content to summarize
    
    Returns:
        A dictionary with the summary and source information
    """
    debug_log(f"=== CREATE_SUMMARY TOOL CALLED ===")
    debug_log(f"Query: '{query}'")
    debug_log(f"Number of documents to summarize: {len(documents)}")
    
    if not client:
        debug_log("Cannot create summary: Anthropic client not initialized")
        return {
            "summary_text": "Summary unavailable: Claude API access is required for summarization.",
            "citations": []
        }
    
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

@mcp.tool()
def search_and_summarize(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a complete search and summarization pipeline.
    
    Args:
        query: The search query
        num_results: Number of search results to use (default: 5)
    
    Returns:
        Dictionary with search results, content, and summary
    """
    debug_log(f"=== SEARCH_AND_SUMMARIZE PIPELINE CALLED ===")
    debug_log(f"Query: '{query}'")
    debug_log(f"Number of results requested: {num_results}")
    
    try:
        # Step 1: Perform search
        debug_log("Step 1: Performing search")
        search_results = web_search(query, num_results)
        
        if "error" in search_results:
            debug_log(f"Error in search: {search_results.get('error')}")
            return {
                "error": search_results.get("error"),
                "summary_text": f"Search failed: {search_results.get('error')}",
                "citations": []
            }
        
        # Step 2: Extract content from each search result
        debug_log("Step 2: Extracting content from search results")
        content_results = []
        
        for result in search_results.get("results", []):
            url = result.get("url")
            if url:
                content = web_fetch(url)
                if "error" not in content:
                    content_results.append(content)
        
        if not content_results:
            debug_log("No content could be extracted from any of the search results")
            return {
                "search_results": search_results,
                "content": [],
                "summary_text": "No content could be extracted from the search results.",
                "citations": []
            }
        
        # Step 3: Rank content
        debug_log("Step 3: Ranking content")
        ranked_content = rank_content(content_results, query)
        
        # Step 4: Create summary
        debug_log("Step 4: Creating summary")
        summary_result = create_summary(query, ranked_content[:min(len(ranked_content), 5)])
        
        result = {
            "search_results": search_results,
            "content": content_results,
            "ranked_content": ranked_content,
            "summary_text": summary_result.get("summary_text", "No summary available"),
            "citations": summary_result.get("citations", [])
        }
        
        return result
        
    except Exception as e:
        debug_log(f"Error in search_and_summarize: {str(e)}")
        return {
            "error": f"Pipeline failed: {str(e)}",
            "summary_text": f"Search pipeline failed: {str(e)}",
            "citations": []
        }


@mcp.tool()
def process_query(query: str, use_claude: bool = True) -> Dict[str, Any]:
    """
    Process a user query to enhance it for search.
    
    Args:
        query: The user's search query
        use_claude: Whether to use Claude to enhance the query
    
    Returns:
        JSON string with original and enhanced query
    """
    debug_log(f"=== PROCESS_QUERY TOOL CALLED ===")
    debug_log(f"Query: '{query}'")
    debug_log(f"Use Claude enhancement: {use_claude}")

    try:
        # Import QueryProcessor from existing code
        from query import QueryProcessor
        
        query_processor = QueryProcessor(claude_client=client if use_claude else None)
        query_analysis = query_processor.process_query(query, use_context=True, use_claude=use_claude)
        
        debug_log(f"Query processing successful. Enhanced query: {query_analysis.get('enhanced_query', query)}")
        return query_analysis
    except Exception as e:
        debug_log(f"Error in process_query: {str(e)}")
        return {
            "query": query,
            "enhanced_query": query,
            "error": str(e)
        }

# Add request/response logging middleware
@mcp.tool(description="Health check to verify server is running")
def health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint to verify server is responding.
    
    Returns:
        Dictionary with server status and timestamp
    """
    debug_log("=== HEALTH_CHECK TOOL CALLED ===")
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "server": "Web Search MCP Server"
    }

# Main execution
if __name__ == "__main__":
    debug_log("Starting MCP Server")
    debug_log("Available tools: web_search, web_fetch, assign_credibility_score, rank_content, create_summary, search_and_summarize, process_query, health_check")
    debug_log("Server ready to accept connections via STDIO transport")
    debug_log("Waiting for client connection...")
    
    try:
        debug_log("Starting MCP transport...")
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        debug_log("Server shutdown requested by user")
    except Exception as e:
        debug_log(f"Server error: {str(e)}")
        import traceback
        debug_log(f"Full traceback: {traceback.format_exc()}")
    finally:
        debug_log("MCP Server shutting down")
        f.close()