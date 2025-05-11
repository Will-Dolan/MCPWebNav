import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

async def main():
    print("Starting Web Navigation Client...")
    
    # Define server parameters for stdio connection
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the current Python interpreter
        args=["server.py"],  # Your server script
        env=os.environ.copy()  # Pass current environment variables
    )
    
    # Connect to the server
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Create a client session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            print("Connected to server successfully")
            
            # List available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            print(f"Available tools: {tool_names}")
            
            # Get user query
            search_query = input("Enter your search query: ") or "Python programming"
            print(f"\nSearching for: {search_query}")
            
            # Option to use navigation tool directly or execute step by step
            use_navigation = input("Use full navigation? (y/n, default: y): ").lower() != "n"
            
            if use_navigation:
                # Use the navigate_web tool to handle the entire process
                print("\nRunning complete web navigation...")
                navigation_results = await session.call_tool("navigate_web", {
                    "query": search_query,
                    "max_depth": 3
                })
                
                if navigation_results.isError:
                    print(f"Error in navigation: {navigation_results.content[0].text if navigation_results.content else 'No content'}")
                else:
                    try:
                        nav_data = json.loads(navigation_results.content[0].text)
                        
                        print("\n===== NAVIGATION RESULTS =====")
                        print(f"Action: {nav_data.get('action')}")
                        print(f"Query: {nav_data.get('query')}")
                        print(f"Depth reached: {nav_data.get('depth_reached')}")
                        print(f"Visited URLs: {len(nav_data.get('visited_urls', []))} pages")
                        
                        if nav_data.get('action') == 'extract':
                            print("\n===== EXTRACTED CONTENT =====")
                            print(nav_data.get('extracted_content'))
                        else:
                            # If we reached max depth without extracting
                            print("\n===== COLLECTED PASSAGES =====")
                            for i, passage in enumerate(nav_data.get('collected_passages', [])):
                                print(f"\nPassage {i+1} (excerpt):")
                                # Print only a brief excerpt of each passage
                                print(passage[:300] + "..." if len(passage) > 300 else passage)
                        
                    except json.JSONDecodeError:
                        print(f"Raw content (not JSON): {navigation_results.content[0].text}")
            else:
                # Execute step-by-step process
                
                # Step 1: Search for URLs
                print("\nStep 1: Searching for relevant URLs...")
                search_results = await session.call_tool("google_search", {"query": search_query})
                
                if search_results.isError:
                    print(f"Error in search: {search_results.content[0].text if search_results.content else 'No content'}")
                    return
                
                search_data = json.loads(search_results.content[0].text)
                if not search_data or "error" in search_data[0]:
                    print(f"Error in search results: {search_data[0].get('error', 'Unknown error')}")
                    return
                
                print(f"Found {len(search_data)} URLs:")
                for i, result in enumerate(search_data):
                    print(f"{i+1}. {result.get('url')}")
                
                # Step 2: Scrape the first URL
                if not search_data:
                    print("No URLs found to scrape.")
                    return
                
                url_to_scrape = search_data[0].get('url')
                print(f"\nStep 2: Scraping webpage: {url_to_scrape}")
                
                scrape_results = await session.call_tool("scrape_webpage", {"url": url_to_scrape})
                
                if scrape_results.isError:
                    print(f"Error in scraping: {scrape_results.content[0].text if scrape_results.content else 'No content'}")
                    return
                
                scrape_data = json.loads(scrape_results.content[0].text)
                if "error" in scrape_data:
                    print(f"Error in scrape results: {scrape_data.get('error')}")
                    return
                
                print(f"Scraped page title: {scrape_data.get('title')}")
                print(f"Content length: {len(scrape_data.get('text_content', ''))} characters")
                print(f"Found {len(scrape_data.get('links', []))} links on the page")
                
                # Step 3: Analyze the content
                print("\nStep 3: Analyzing content...")
                
                # Prepare passages and links for analysis
                passages = [scrape_data.get('text_content', '')]
                links = scrape_data.get('links', [])
                
                analysis_results = await session.call_tool("analyze_content", {
                    "query": search_query,
                    "passages": passages,
                    "links": links
                })
                
                if analysis_results.isError:
                    print(f"Error in analysis: {analysis_results.content[0].text if analysis_results.content else 'No content'}")
                    return
                
                analysis_data = json.loads(analysis_results.content[0].text)
                
                print("\n===== ANALYSIS RESULTS =====")
                print(f"Decision: {analysis_data.get('action')}")
                
                if analysis_data.get('action') == 'explore':
                    print("\nSuggested links to explore:")
                    for i, link in enumerate(analysis_data.get('links_to_explore', [])):
                        print(f"{i+1}. {link.get('text', 'No text')}: {link.get('url')}")
                else:
                    print("\nRelevant content:")
                    print(analysis_data.get('relevant_content'))

if __name__ == "__main__":
    asyncio.run(main())