"""
Server implementation that integrates both server.py and backend_api.py functionality
with a complete pipeline for web search and content extraction.
"""
from mcp.server import FastMCP
import requests
from bs4 import BeautifulSoup
import anthropic
import os
import json
import datetime
import torch
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Import necessary modules if available
try:
    from query import QueryProcessor
    from synthesis.source_ranking import load_tokenizer_and_model, embed_documents, mean_pooling
    from synthesis.create_response import summarize_documents
except ImportError:
    print("Warning: Some custom modules could not be imported. Some functionality may be limited.")

load_dotenv()

# Initialize the MCP server
mcp = FastMCP("Web Search Server")

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Initialize QueryProcessor with the Claude client
query_processor = QueryProcessor(claude_client=client)

# Initialize debug logging
debug_file_path = os.path.join(os.getcwd(), "server_debug.txt")
f = open(debug_file_path, "w")

def debug_log(message):
    """Add timestamped messages to the debug log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"{timestamp} - {message}\n")
    f.flush()

# Log at startup
debug_log(f"Server starting up. Debug log at: {debug_file_path}")
debug_log(f"Current working directory: {os.getcwd()}")

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
        # Process the query using QueryProcessor
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
    """Perform mean pooling on the model output."""
    # Extract last hidden state from model
    token_embeddings = model_output.last_hidden_state 
    # Expand attention mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Calculate mean embeddings
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)  # Prevent division by zero
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings

# Function to load tokenizer and model
def load_tokenizer_and_model(name):
    """Load the tokenizer and model for embeddings."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return tokenizer, model

# Function to make embeddings of documents
def embed_documents(documents, tokenizer, model):
    """Create embeddings for documents."""
    # Tokenize the documents
    encoded_input = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Disable gradient calculations
    with torch.no_grad():
        model_output = model(**encoded_input)
        vectors = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return vectors

# Function to assign credibility score to URLs
def assign_credibility_score(urls):
    """Assign a credibility score to a list of URLs using Claude."""
    debug_log(f"assign_credibility_score called with {len(urls)} URLs")
    
    try:
        # Create prompt for Claude
        urls_text = "\n".join(urls)
        prompt = f"Assign a credibility score (0.00-1.00) to the following URLs based on the credibility of the website, where 1.00 is most credible:\n{urls_text}\n\n"
        prompt += "Return a JSON object with the URL as the key and the score as the value."
        
        # System prompt for Claude
        system_prompt = "You are a helpful assistant that measures the credibility scores of URLs. Only respond with the scores in JSON format."
        
        # Call Claude API
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
        
        # Parse JSON response
        credibility_scores = json.loads(response_text)
        debug_log(f"Successfully assigned credibility scores to {len(credibility_scores)} URLs")
        
        return credibility_scores
    except Exception as e:
        debug_log(f"Error in assign_credibility_score: {str(e)}")
        return {}  # Return empty dict if assignment fails

def conflict_prompt(higher_text: str, lower_text: str) -> str:
    """Create a prompt to detect conflicts between two texts."""
    return (
        "You are a helpful assistant that determines if two documents contain conflicting factual information. "
        "Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise.\n\n"
        f"Higher-priority document:\n{higher_text}\n\n"
        f"Lower-priority document:\n{lower_text}\n\n"
        "Do these two documents conflict? Answer 'Yes' or 'No'."
    )

def has_conflict(higher_text: str, lower_text: str) -> bool:
    """Determine if two texts contain conflicting information."""
    debug_log(f"Checking for conflicts between documents")
    
    try:
        # Create prompt for Claude
        prompt = conflict_prompt(higher_text, lower_text)
        
        # System prompt for conflict detection
        system_prompt = "You are a helpful assistant that determines if two documents contain conflicting factual information. Given a higher-priority document and a lower-priority document, respond with 'Yes' if they conflict, and 'No' otherwise."
        
        # Call Claude API
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
        return False  # Default to no conflict if the check fails

def resolve_conflicts(docs):
    """Filter out conflicting documents, keeping the most credible ones."""
    debug_log(f"resolve_conflicts called with {len(docs)} documents")
    
    try:
        # Sort by credibility (highest first)
        docs_by_credibility = sorted(docs, key=lambda x: x.get("credibility", 0), reverse=True)
        
        filtered = []
        for doc in docs_by_credibility:
            conflict_found = False
            for kept in filtered:
                if has_conflict(kept["content"], doc["content"]):
                    debug_log(f"Excluding document due to conflict with higher credibility source")
                    conflict_found = True
                    break
            if not conflict_found:
                filtered.append(doc)
                
        debug_log(f"After conflict resolution: {len(filtered)} documents remain")
        return filtered
    except Exception as e:
        debug_log(f"Error in resolve_conflicts: {str(e)}")
        return docs  # Return original docs if conflict resolution fails

def rank_sources(docs, query):
    """Rank sources based on relevance to the query and their credibility."""
    debug_log(f"rank_sources called with query: {query}, {len(docs)} documents")
    try:
        # Load model
        tokenizer, model = load_tokenizer_and_model('sentence-transformers/all-MiniLM-L6-v2')
        
        # Extract text content from documents
        doc_texts = [doc["content"] for doc in docs]
        
        # Generate embeddings
        doc_embeddings = torch.vstack([embed_documents([text], tokenizer, model) for text in doc_texts])
        query_embedding = embed_documents([query], tokenizer, model)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get URLs for credibility scoring
        urls = [doc.get("url", "") for doc in docs]
        
        # Get credibility scores
        credibility_scores = assign_credibility_score(urls)
        
        # Add similarity and credibility scores to documents
        for i, doc in enumerate(docs):
            doc["similarity"] = float(similarities[i])
            url = doc.get("url", "")
            doc["credibility"] = credibility_scores.get(url, 0.5)  # Default to 0.5 if not found
        
        # Resolve conflicts between documents
        docs_without_conflicts = resolve_conflicts(docs)
        
        # Sort documents by similarity score
        ranked_docs = sorted(docs_without_conflicts, key=lambda x: x.get("similarity", 0), reverse=True)
        
        debug_log(f"Successfully ranked {len(ranked_docs)} documents")
        return ranked_docs
    except Exception as e:
        debug_log(f"Error in rank_sources: {str(e)}")
        return docs  # Return original docs if ranking fails

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
        from googlesearch import search
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
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract main content (simplified approach)
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
        
        # Extract text, similarity scores, and credibility scores
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
                
                # Create citation data structure
                citation_sources.append({
                    "index": i+1,
                    "url": doc.get('url', 'Unknown URL'),
                    "title": doc.get('title', 'Unknown Title'),
                    "similarity": doc.get('similarity', 0.0),
                    "credibility": doc.get('credibility', 0.5),
                    "snippet": doc.get('content', '')[:200] + "..." if doc.get('content') else ""
                })
        
        # Create the prompt for Claude
        prompt = ""
        for i, text in enumerate(doc_texts):
            prompt += f"Document {i+1} (sim score = {similarity_scores[i]:.2f}, credibility score = {credibility_scores[i]:.2f}):\n{text}\n\n"
        
        prompt += f"Query: {query}\n\n"
        prompt += "Answer the query by summarizing information from the documents above."
        
        debug_log("Created prompt for summarization")
        
        # System prompt for Claude
        system_prompt = """
        You are a helpful assistant that summarizes multiple documents in order to answer a user's query. The similarity scores
        and credibility scores for each document are also provided. Answer the query to the best of your understanding. Do not 
        include external information or make up facts. Only include the answer.
        """
        
        # Call Claude API
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
        
        # Return both summary and citation information separately
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
def complete_search_pipeline(query: str, num_results: int = 5) -> dict:
    """
    Run the complete search pipeline: search, extract content, rank sources, and create summary.
    Now uses the enhanced query directly from get_chat_response.
    
    Args:
        query: The search query (already enhanced)
        num_results: Number of search results to use (default: 5)
    
    Returns:
        Dictionary with all search results and summary
    """
    debug_log(f"complete_search_pipeline called with query: {query}, num_results: {num_results}")
    
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
        
        # Step 2: Extract content from each search result
        debug_log("Step 2: Extracting content from search results")
        content_results = []
        
        for result in search_results:
            url = result.get("url")
            if url:
                debug_log(f"Scraping content from: {url}")
                content = scrape_webpage(url)
                if "error" not in content:
                    content_results.append(content)
        
        if not content_results:
            debug_log("No content could be extracted from any of the search results")
            return {
                "search_results": search_results,
                "content": [],
                "ranked_content": [],
                "summary_text": "No content could be extracted from the search results.",
                "citations": []
            }
        
        # Step 3: Rank sources based on relevance to the query and credibility
        debug_log("Step 3: Ranking sources by similarity and credibility")
        ranked_content = rank_sources(content_results, query)
        
        # Step 4: Create summary from ranked content
        debug_log("Step 4: Creating summary")
        summary_result = create_summary(query, ranked_content[:max(len(ranked_content), 5)])  # Use top 5 sources for summary
        
        # Prepare final result
        result = {
            "search_results": search_results,
            "content": content_results,
            "ranked_content": ranked_content,
            "summary_text": summary_result.get("summary_text", "No summary available"),
            "citations": summary_result.get("citations", [])
        }
        
        # Save results to file
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
def get_chat_response(query: str, chat_history: list = None) -> dict:
    """
    Get a response for a user query with chat history context.
    Uses QueryProcessor to enhance queries before searching.
    
    Args:
        query: The user's query
        chat_history: List of previous queries and responses (optional)
    
    Returns:
        Dictionary with search results and response
    """
    debug_log(f"get_chat_response called with query: {query}, history length: {len(chat_history) if chat_history else 0}")
    
    try:
        # Process the query to enhance it using QueryProcessor
        debug_log(f"Original query: {query}")
        use_claude_for_query = True  # This can be a config parameter
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

        # Run the search pipeline with the enhanced query
        search_results = complete_search_pipeline(enhanced_query)
        
        # Add query information to search results
        search_results["original_query"] = query
        search_results["enhanced_query"] = enhanced_query
        
        # If search has an error, return it
        if "error" in search_results:
            debug_log(f"Search error: {search_results['error']}")
            return {
                "query": query,
                "enhanced_query": enhanced_query,
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
        
        # Get the summary text from the search results
        summary_text = search_results.get("summary_text", "No summary available")
        citations = search_results.get("citations", [])
        
        # Add search results to the response
        response = {
            "query": query,
            "enhanced_query": enhanced_query,
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
            "response": f"Sorry, I encountered an error: {str(e)}",
            "search_results": {},
            "citations": []
        }

if __name__ == "__main__":
    debug_log("Starting server.py")
    mcp.run(transport='stdio')