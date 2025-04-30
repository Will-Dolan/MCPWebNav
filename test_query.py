import json
from mcp.query import QueryProcessor

def test_query_processor():
    print("Testing QueryProcessor...")
    processor = QueryProcessor()
    
    # Test basic query
    test_query = "What is quantum computing?"
    result = processor.process_query(test_query)
    print(f"Original query: {result['original_query']}")
    print(f"Enhanced query: {result['enhanced_query']}")
    print(f"Search query: {result['search_query']}")
    print(f"Entity data: {result['entity_data']}")
    print(f"Search strategy: {result['search_strategy']}")
    print(f"Refinement suggestions: {result['refinement_suggestions']}")
    print()
    
    # Test follow-up query
    follow_up = "How is it used in cryptography?"
    result2 = processor.process_query(follow_up)
    print(f"Follow-up query: {follow_up}")
    print(f"Enhanced query with context: {result2['enhanced_query']}")
    print()
    
    # Test a specific domain query
    domain_query = "Show me research papers about AI ethics"
    result3 = processor.process_query(domain_query, use_context=False)
    print(f"Domain query: {domain_query}")
    print(f"Entity data: {result3['entity_data']}")
    print(f"Search strategy: {result3['search_strategy']}")
    print(f"Search query: {result3['search_query']}")
    print()
    
    # Test formatted output for passing to downstream components
    print("Sample output for downstream components:")
    print(json.dumps({
        "query": result["enhanced_query"],
        "query_embedding": result["query_embedding"],
        "search_strategy": result["search_strategy"]
    }, indent=2))
    
    return result

if __name__ == "__main__":
    # Test query processor individually
    result = test_query_processor()