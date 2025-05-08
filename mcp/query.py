import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class QueryProcessor:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer, self.model = self.load_tokenizer_and_model(model_name)
        self.conversation_history = []
        
    def load_tokenizer_and_model(self, name):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        return tokenizer, model
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    def embed_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embedding
    
    def extract_entities(self, query):
        # Simple entity extraction based on query keywords
        # In a production system, you would use spaCy or similar NLP libraries here
        
        # Define common entity patterns
        entities = {
            'what': 'definition',
            'who': 'person',
            'where': 'location',
            'when': 'time',
            'why': 'reason',
            'how': 'method'
        }
        
        query_lower = query.lower()
        query_type = 'general'
        
        # Determine query type
        for keyword, entity_type in entities.items():
            if query_lower.startswith(keyword):
                query_type = entity_type
                break
                
        # Extract important keywords (simple approach)
        ignore_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'for', 'on', 'with'}
        important_words = [word for word in query_lower.split() if word not in ignore_words]
        
        return {
            'query_type': query_type,
            'keywords': important_words,
            'full_query': query
        }
    
    def format_search_query(self, entity_data):
        # Format the search query based on entity type
        query_type = entity_data['query_type']
        keywords = entity_data['keywords']
        
        # Join all keywords with '+' for search engine compatibility
        search_query = '+'.join(keywords)
        
        # Add specific modifiers based on query type
        if query_type == 'definition':
            search_query += '+meaning+definition'
        elif query_type == 'person':
            search_query += '+biography+who'
        elif query_type == 'location':
            search_query += '+place+location'
        
        return search_query
    
    def determine_search_strategy(self, query, entity_data):
        # Determine the appropriate search strategy based on query analysis
        query_type = entity_data['query_type']
        keywords = entity_data['keywords']
        
        # Default strategy
        strategy = {
            'depth': 2,  # How many links deep to follow
            'max_sources': 5,  # Maximum number of sources to consult
            'domain_focus': 'general',  # Focus on specific domains or general search
            'follow_links': True  # Whether to follow links from initial results
        }
        
        # Adjust strategy based on query type
        if query_type == 'definition':
            strategy['domain_focus'] = 'encyclopedic'
            strategy['depth'] = 1  # Definitions usually need less depth
            
        elif query_type == 'person':
            strategy['domain_focus'] = 'biographical'
            
        elif query_type == 'location':
            strategy['domain_focus'] = 'geographical'
            
        # Check for academic or scientific queries
        academic_terms = ['research', 'study', 'paper', 'scientific', 'academic', 'journal']
        if any(term in ' '.join(keywords) for term in academic_terms):
            strategy['domain_focus'] = 'academic'
            strategy['depth'] = 3  # Academic queries often need deeper exploration
            
        return strategy
    
    def handle_context(self, query):
        # Handle context from previous queries
        if not self.conversation_history:
            # No previous context, just process the query directly
            return query
            
        # Get the most recent conversation
        recent_query = self.conversation_history[-1]['query']
        recent_entities = self.conversation_history[-1]['entities']
        
        # Check if the current query is a follow-up
        follow_up_indicators = ['it', 'this', 'that', 'they', 'them', 'their', 'he', 'she', 'his', 'her']
        query_words = query.lower().split()
        
        is_follow_up = any(word in follow_up_indicators for word in query_words)
        
        if is_follow_up:
            # This is likely a follow-up question, so incorporate previous context
            # For a simple approach, we'll just append the context keywords
            context_keywords = ' '.join(recent_entities['keywords'])
            enhanced_query = f"{query} regarding {context_keywords}"
            return enhanced_query
            
        return query
    
    def suggest_refinements(self, query, entity_data):
        # Suggest query refinements to help the user get better results
        query_type = entity_data['query_type']
        keywords = entity_data['keywords']
        
        suggestions = []
        
        # Add specificity for general queries
        if query_type == 'general' and len(keywords) < 3:
            suggestions.append(f"{query} detailed explanation")
            
        # Add recent information suggestion
        suggestions.append(f"{query} recent developments")
        
        # Add comparison suggestion if appropriate
        if query_type == 'definition':
            suggestions.append(f"{query} compared to alternatives")
            
        return suggestions
    
    def process_query(self, query, use_context=True):
        # Main method to process a query
        
        # Handle context from previous conversations if enabled
        if use_context:
            enhanced_query = self.handle_context(query)
        else:
            enhanced_query = query
            
        # Extract entities and query type
        entity_data = self.extract_entities(enhanced_query)
        
        # Format the search query
        search_query = self.format_search_query(entity_data)
        
        # Determine search strategy
        strategy = self.determine_search_strategy(enhanced_query, entity_data)
        
        # Generate query refinement suggestions
        refinement_suggestions = self.suggest_refinements(enhanced_query, entity_data)
        
        # Create embeddings for the query (for later similarity comparison)
        query_embedding = self.embed_text([enhanced_query]).tolist()
        
        # Store this query in conversation history
        self.conversation_history.append({
            'query': enhanced_query,
            'entities': entity_data,
            'embedding': query_embedding
        })
        
        # Prepare final output
        query_analysis = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'search_query': search_query,
            'entity_data': entity_data,
            'search_strategy': strategy,
            'refinement_suggestions': refinement_suggestions,
            'query_embedding': query_embedding
        }
        
        return query_analysis


def main():
    # Example usage
    processor = QueryProcessor()
    
    # Process a sample query
    query = "What is quantum computing?"
    result = processor.process_query(query)
    
    # Print the result
    print(json.dumps(result, indent=2))
    
    # Process a follow-up query
    follow_up = "How is it used in cryptography?"
    result2 = processor.process_query(follow_up)
    
    # Print the follow-up result
    print(json.dumps(result2, indent=2))


if __name__ == '__main__':
    main()