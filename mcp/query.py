import json
import os
import anthropic

class QueryProcessor:
    def __init__(self, claude_client=None):
        self.conversation_history = []
        self.client = claude_client
        
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
    
    def enhance_query_with_claude(self, query):
        """
        Use Claude to enhance the search query by extracting key concepts
        and reformulating it for better search results.
        
        Args:
            query: The original user query
            
        Returns:
            Enhanced query string
        """
        if not self.client:
            # If Claude client isn't available, return original query
            return query
            
        prompt = f"""
        Your task is to reformulate the following search query to make it more effective for web search.
        
        Original Query: "{query}"
        
        Please analyze this query and:
        1. Identify the core information need
        2. Extract key concepts and entities
        3. Reformulate as a search-engine optimized query
        4. Use "+" between terms for search engine compatibility
        
        For example:
        - "What's the capital of France?" → "capital+of+France+Paris"
        - "How do birds migrate?" → "bird+migration+how+navigation+seasons"
        
        Reply with ONLY the reformulated query, no other text.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=100,
                temperature=0,
                system="You are an AI assistant specialized in optimizing search queries. Your task is to reformulate user queries to be more effective for search engines.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            enhanced_query = response.content[0].text.strip()
            return enhanced_query
            
        except Exception as e:
            # If there's an error with Claude, return the original query
            return query
    
    def process_query(self, query, use_context=True, use_claude=True):
        """
        Main method to process a query with optional Claude enhancement
        
        Args:
            query: The user's query
            use_context: Whether to use conversation context
            use_claude: Whether to use Claude for query enhancement
            
        Returns:
            Query analysis dictionary
        """
        # Handle context from previous conversations if enabled
        if use_context:
            enhanced_query = self.handle_context(query)
        else:
            enhanced_query = query
        
        # Use Claude to further enhance the query if requested
        if use_claude and self.client:
            claude_enhanced_query = self.enhance_query_with_claude(enhanced_query)
            
            # Prefer Claude's enhancement if available and different from original
            if claude_enhanced_query and claude_enhanced_query != enhanced_query:
                enhanced_query = claude_enhanced_query
                
                # Extract entities from Claude's enhanced query for better entity analysis
                entity_data = self.extract_entities(enhanced_query)
            else:
                # If Claude didn't change the query, use our standard entity extraction
                entity_data = self.extract_entities(enhanced_query)
        else:
            # Use standard entity extraction if not using Claude
            entity_data = self.extract_entities(enhanced_query)
        
        # Format the search query - if Claude provided a "+" formatted query, use it directly
        if use_claude and "+" in enhanced_query and not " " in enhanced_query.strip():
            search_query = enhanced_query  # Claude already formatted it for search
        else:
            search_query = self.format_search_query(entity_data)
        
        # Determine search strategy
        strategy = self.determine_search_strategy(enhanced_query, entity_data)
        
        # Generate query refinement suggestions
        refinement_suggestions = self.suggest_refinements(enhanced_query, entity_data)
        
        # Store this query in conversation history
        self.conversation_history.append({
            'query': enhanced_query,
            'entities': entity_data
        })
        
        # Prepare final output
        query_analysis = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'search_query': search_query,
            'entity_data': entity_data,
            'search_strategy': strategy,
            'refinement_suggestions': refinement_suggestions,
            'claude_enhanced': use_claude and self.client is not None
        }
        
        return query_analysis