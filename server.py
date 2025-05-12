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
        
        # Step 2: Extract content from each search result in parallel
        debug_log("Step 2: Extracting content from search results using parallel processing")
        urls_to_scrape = [result.get("url") for result in search_results if result.get("url")]
        
        # Use ThreadPoolExecutor to fetch content in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            scrape_tasks = [executor.submit(scrape_webpage, url) for url in urls_to_scrape]
            
            content_results = []
            for task in concurrent.futures.as_completed(scrape_tasks):
                content = task.result()
                if "error" not in content:
                    content_results.append(content)
                    debug_log(f"Successfully scraped content from: {content.get('url')}")
        
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
        
        result = {
            "search_results": search_results,
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

        search_results = complete_search_pipeline(enhanced_query)
        search_results["original_query"] = query
        search_results["enhanced_query"] = enhanced_query
        
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
        
        summary_text = search_results.get("summary_text", "No summary available")
        citations = search_results.get("citations", [])
        
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