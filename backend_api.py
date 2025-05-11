import os
from dotenv import load_dotenv
import anthropic
import torch
from sklearn.metrics.pairwise import cosine_similarity

from mcp.query import QueryProcessor
from mcp.get_urls import query_common_crawl, query_to_url_patterns
from mcp.extract import extract_info_from_links
from synthesis.source_ranking import load_tokenizer_and_model, embed_documents, mean_pooling
from synthesis.create_response import summarize_documents

load_dotenv()

# Local wrapper for get_urls since teammates didn't define `search_urls()`
def search_urls(query, max_urls=5):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    url_patterns = query_to_url_patterns(client, query)

    all_urls = []
    for pattern in url_patterns:
        matches = query_common_crawl(pattern)
        all_urls.extend(matches)
        if len(all_urls) >= max_urls:
            break

    return all_urls[:max_urls]


# Direct integration of source ranking logic without file dependency
def rank_sources(docs, query):
    # TODO: handle this outside of the function so we don't do every time
    tokenizer, model = load_tokenizer_and_model('sentence-transformers/all-MiniLM-L6-v2')
    doc_texts = [doc["text"] for doc in docs]
    doc_embeddings = torch.vstack([embed_documents(text, tokenizer, model) for text in doc_texts])
    query_embedding = embed_documents([query], tokenizer, model)

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked_indices = similarities.argsort()[::-1].tolist()

    ranked_docs = []
    for idx in ranked_indices:
        ranked_docs.append(docs[idx])

    return ranked_docs


def query_web_agent(user_query: str, api_key: str) -> dict:
    # Step 1: Analyze query and enhance it
    processor = QueryProcessor()
    query_data = processor.process_query(user_query)

    search_query = query_data["search_query"]
    strategy = query_data["search_strategy"]
    refinement_suggestions = query_data["refinement_suggestions"]

    # Step 2: Get URLs from Common Crawl
    urls = search_urls(search_query, max_urls=strategy.get("max_sources", 5))

    # Step 3: Extract info from URLs using Claude
    summaries = extract_info_from_links(urls)
    documents = [{"url": url, "text": summary} for url, summary in zip(urls, summaries)]

    if not documents:
        return {
            "query": user_query,
            "answer": "Sorry, I couldn't find any useful information.",
            "citations": [],
            "visited_links": [],
            "related_questions": refinement_suggestions
        }

    # Step 4: Rank documents based on similarity
    ranked_docs = rank_sources(documents, user_query)

    # Step 5: Summarize using Claude
    answer = summarize_documents(ranked_docs, user_query)

    return {
        "query": user_query,
        "answer": answer,
        "citations": [],  # Skipped for now
        "visited_links": [doc["url"] for doc in ranked_docs],
        "related_questions": refinement_suggestions
    }