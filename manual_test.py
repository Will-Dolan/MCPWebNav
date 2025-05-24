#!/usr/bin/env python3
"""
Manual test by directly calling server functions
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the server module directly
import mcp_server

def test_complete_pipeline():
    """Test the complete search and summarization pipeline"""
    query = "what is quantum computing"
    print(f"[MANUAL] Testing complete pipeline for query: '{query}'")
    
    # Test health check
    print("\n=== Health Check ===")
    result = mcp_server.health_check()
    print(f"✓ Server status: {result['status']}")
    
    # Test the complete search and summarize pipeline
    print(f"\n=== Complete Search & Summarize Pipeline ===")
    try:
        pipeline_result = mcp_server.search_and_summarize(query, 3)
        
        if "error" in pipeline_result:
            print(f"✗ Pipeline error: {pipeline_result['error']}")
            return
        
        # Display search results
        search_results = pipeline_result.get('search_results', {}).get('results', [])
        print(f"✓ Found {len(search_results)} search results:")
        for i, res in enumerate(search_results, 1):
            print(f"  {i}. {res.get('url', 'No URL')}")
        
        # Display content extraction
        content_results = pipeline_result.get('content', [])
        print(f"✓ Successfully scraped {len(content_results)} pages:")
        for i, content in enumerate(content_results, 1):
            title = content.get('title', 'No Title')
            content_length = len(content.get('content', ''))
            print(f"  {i}. {title} ({content_length} chars)")
        
        # Display ranking
        ranked_content = pipeline_result.get('ranked_content', [])
        if ranked_content:
            print(f"✓ Content ranked by relevance:")
            for i, doc in enumerate(ranked_content[:3], 1):
                similarity = doc.get('similarity', 0)
                credibility = doc.get('credibility', 0)
                title = doc.get('title', 'No Title')
                print(f"  {i}. {title}")
                print(f"     Relevance: {similarity:.3f}, Credibility: {credibility:.3f}")
        
        # Display summary
        summary_text = pipeline_result.get('summary_text', '')
        if summary_text and summary_text != "No summary available":
            print(f"\n=== Generated Summary ===")
            print(f"Summary length: {len(summary_text)} characters")
            print(f"Summary preview: {summary_text[:300]}...")
            
            # Display citations
            citations = pipeline_result.get('citations', [])
            if citations:
                print(f"\n=== Citations ({len(citations)} sources) ===")
                for i, citation in enumerate(citations, 1):
                    print(f"{i}. {citation.get('title', 'No Title')}")
                    print(f"   URL: {citation.get('url', 'No URL')}")
                    print(f"   Relevance: {citation.get('similarity', 0):.3f}")
                    print(f"   Credibility: {citation.get('credibility', 0):.3f}")
        else:
            print("✗ No summary generated")
        
        print(f"\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

def test_individual_tools():
    """Test individual tools separately"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL TOOLS")
    print("="*50)
    
    query = "artificial intelligence"
    
    # Test web search
    print("\n=== 1. Web Search ===")
    search_result = mcp_server.web_search(query, 2)
    urls = [r['url'] for r in search_result.get('results', [])]
    print(f"Found {len(urls)} URLs")
    
    if urls:
        # Test web fetch
        print("\n=== 2. Web Fetch ===")
        documents = []
        for i, url in enumerate(urls[:2], 1):
            print(f"Fetching {i}: {url}")
            fetch_result = mcp_server.web_fetch(url)
            if 'content' in fetch_result:
                documents.append(fetch_result)
                print(f"  ✓ {len(fetch_result['content'])} chars")
            else:
                print(f"  ✗ Error: {fetch_result.get('error', 'Unknown')}")
        
        if documents:
            # Test ranking
            print("\n=== 3. Content Ranking ===")
            ranked_docs = mcp_server.rank_content(documents, query)
            for i, doc in enumerate(ranked_docs, 1):
                similarity = doc.get('similarity', 0)
                credibility = doc.get('credibility', 0)
                print(f"  {i}. Sim: {similarity:.3f}, Cred: {credibility:.3f}")
            
            # Test summary creation
            print("\n=== 4. Summary Creation ===")
            summary_result = mcp_server.create_summary(query, ranked_docs)
            summary = summary_result.get('summary_text', '')
            citations = summary_result.get('citations', [])
            
            if summary and "Failed to create summary" not in summary:
                print(f"✓ Summary created: {len(summary)} chars")
                print(f"✓ Citations: {len(citations)} sources")
                print(f"Preview: {summary[:200]}...")
            else:
                print(f"✗ Summary failed: {summary}")

def test_tools_directly():
    """Main test function"""
    test_complete_pipeline()
    test_individual_tools()

if __name__ == "__main__":
    test_tools_directly()