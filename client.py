"""
Client for connecting to the server.py MCP server.
This client implements a continuous chat interface with automatic saving of search results.
"""
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import argparse
import datetime

async def main():
    parser = argparse.ArgumentParser(description='Web Search Chat Client')
    parser.add_argument('--save_dir', type=str, default='chat_history',
                       help='Directory to save chat history (default: chat_history)')
    args = parser.parse_args()
    
    print("Starting Web Search Chat Client...")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define server parameters for stdio connection
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the current Python interpreter
        args=["server.py"],  # The server script
        env=os.environ.copy()  # Pass current environment variables
    )
    
    # Create a filename for this chat session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"chat_session_{timestamp}.json"
    session_path = os.path.join(args.save_dir, session_filename)
    
    # Initialize chat history
    chat_history = []
    
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
            print(f"Available tools: {', '.join(tool_names)}")
            
            print("\n===== WEB SEARCH CHAT =====")
            print("Type your questions and get answers from the web.")
            print("Type 'exit', 'quit', or 'bye' to end the chat.")
            print("All chat history and search results are automatically saved.")
            
            # Start chat loop
            while True:
                # Get the user's query
                search_query = input("\nYou: ")
                
                # Check for exit commands
                if search_query.lower() in ['exit', 'quit', 'bye']:
                    print("\nEnding chat session.")
                    break
                
                if not search_query.strip():
                    print("Please enter a question.")
                    continue
                
                print("\nSearching and generating response... Please wait...")
                
                try:
                    # Call the get_chat_response tool with the query and chat history
                    response = await session.call_tool("get_chat_response", {
                        "query": search_query,
                        "chat_history": chat_history
                    })
                    
                    if response.isError:
                        print(f"Error: {response.content[0].text if response.content else 'No content'}")
                        continue
                    
                    # Parse the JSON response
                    result_data = json.loads(response.content[0].text)
                    
                    # Display the assistant's response
                    assistant_response = result_data.get("response", "Sorry, I couldn't find an answer.")
                    print(f"\nAssistant: {assistant_response}")
                    
                    # Add to chat history
                    chat_entry = {
                        "query": search_query,
                        "response": assistant_response,
                        "search_results": result_data.get("search_results", {})
                    }
                    chat_history.append(chat_entry)
                    
                    # Save after each exchange
                    with open(session_path, 'w') as f:
                        json.dump({
                            "timestamp": timestamp,
                            "chat_history": chat_history
                        }, f, indent=2)
                    
                except Exception as e:
                    print(f"Error in chat exchange: {str(e)}")
            
            print(f"\nChat session saved to {session_path}")

if __name__ == "__main__":
    asyncio.run(main())