import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from collections import defaultdict

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
                    
                    # Handle different action types
                    if nav_data.get('action') == 'extract':
                        print("\n===== EXTRACTED CONTENT =====")
                        if 'summary' in nav_data and nav_data['summary']:
                            print("\nSUMMARY:")
                            print(nav_data['summary'])
                        
                        # Group extracted paragraphs by URL
                        paragraphs_by_url = defaultdict(list)
                        for para_data in nav_data.get('extracted_paragraphs', []):
                            url = para_data.get('source', {}).get('url', 'unknown')
                            title = para_data.get('source', {}).get('title', 'Unknown Source')
                            paragraphs_by_url[url].append({
                                'title': title,
                                'paragraph': para_data.get('paragraph', '')
                            })
                        
                        # Display paragraphs grouped by URL
                        print("\nEXTRACTED PARAGRAPHS BY SOURCE:")
                        for url, paragraphs in sorted(paragraphs_by_url.items()):
                            print(f"\nSource: {paragraphs[0]['title']}")
                            print(f"URL: {url}")
                            print("\nRELEVANT PARAGRAPHS:")
                            for i, para in enumerate(paragraphs):
                                print(f"\n[{i+1}] {para['paragraph']}")
                        
                        if 'reasoning' in nav_data and nav_data['reasoning']:
                            print("\nREASONING:")
                            print(nav_data['reasoning'])
                    
                    elif nav_data.get('action') == 'completed_with_extractions':
                        print("\n===== EXTRACTED CONTENT =====")
                        if not nav_data.get('extracted_content'):
                            print("No relevant content was extracted.")
                        else:
                            # Group extracted paragraphs by URL
                            paragraphs_by_url = defaultdict(list)
                            for para_data in nav_data.get('extracted_content', []):
                                url = para_data.get('source', {}).get('url', 'unknown')
                                title = para_data.get('source', {}).get('title', 'Unknown Source')
                                paragraphs_by_url[url].append({
                                    'title': title,
                                    'paragraph': para_data.get('paragraph', '')
                                })
                            
                            # Display paragraphs grouped by URL
                            print("\nEXTRACTED PARAGRAPHS BY SOURCE:")
                            for url, paragraphs in sorted(paragraphs_by_url.items()):
                                print(f"\nSource: {paragraphs[0]['title']}")
                                print(f"URL: {url}")
                                print("\nRELEVANT PARAGRAPHS:")
                                for i, para in enumerate(paragraphs):
                                    print(f"\n[{i+1}] {para['paragraph']}")
                                
                        # Print visited pages summary
                        print("\n===== SOURCES VISITED =====")
                        for i, url in enumerate(nav_data.get('visited_urls', [])):
                            print(f"{i+1}. {url}")
                    
                    else:  # max_depth_reached
                        print("\n===== COLLECTED PASSAGES =====")
                        print("No relevant content was extracted during navigation.")
                        print("Here are the pages that were visited:")
                        
                        # Group passages by URL
                        urls_visited = {}
                        for passage in nav_data.get('collected_passages', []):
                            url = passage.get('url', 'unknown')
                            title = passage.get('title', 'No title')
                            urls_visited[url] = title
                        
                        # Display visited URLs in sorted order
                        for i, (url, title) in enumerate(sorted(urls_visited.items())):
                            print(f"\n{i+1}. {title}")
                            print(f"   URL: {url}")
                    
                    # Ask if user wants to save results to a file
                    save_option = input("\nDo you want to save these results to a file? (y/n): ")
                    if save_option.lower() == 'y':
                        filename = input("Enter filename to save results (default: results.json): ") or "results.json"
                        with open(filename, 'w') as f:
                            json.dump(nav_data, f, indent=2)
                            print(f"Results saved to {filename}")
                    
                except json.JSONDecodeError:
                    print(f"Raw content (not JSON): {navigation_results.content[0].text}")
                except Exception as e:
                    print(f"Error processing results: {str(e)}")
                    print(f"Raw response: {navigation_results.content[0].text if navigation_results.content else 'No content'}")

if __name__ == "__main__":
    asyncio.run(main())