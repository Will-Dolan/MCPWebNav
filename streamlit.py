"""
Combined MCP server and Streamlit interface.
Starts the MCP server in a parallel process and displays results in Streamlit.
"""
import streamlit as st
import os
import sys
import asyncio
import json
import datetime
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

chat_history = []
save_dir = 'chat_history'

def start_mcp_server():
    """Start the MCP server in a separate process"""
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    return server_process

async def query_mcp_server(query, follow_links=True):
    """
    Query the MCP server and return the response
    
    Args:
        query: The user's query
        follow_links: Whether to follow links from search results or just use direct results
    """
    global chat_history
    
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the current Python interpreter
        args=["server.py"],      
        env=os.environ.copy()    # Pass current environment variables
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Use get_chat_response with the follow_links parameter
            response = await session.call_tool("get_chat_response", {
                "query": query,
                "chat_history": chat_history,
                "follow_links": follow_links
            })
            
            if response.isError:
                raise Exception(f"Error: {response.content[0].text if response.content else 'No content'}")
            
            result_data = json.loads(response.content[0].text)
            
            # Add to chat history
            chat_entry = {
                "query": query,
                "response": result_data.get("response", ""),
                "search_results": result_data.get("search_results", {}),
                "citations": result_data.get("citations", []),
                "follow_links_enabled": result_data.get("follow_links_enabled", follow_links)
            }
            chat_history.append(chat_entry)
            
            # Save chat history
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_filename = f"chat_session_{timestamp}.json"
            session_path = os.path.join(save_dir, session_filename)
            
            with open(session_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "chat_history": chat_history
                }, f, indent=2)
            
            return result_data

def run_async_query(query, follow_links):
    """Run the async query in a synchronous context for Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(query_mcp_server(query, follow_links))
    loop.close()
    return result

def main():
    st.set_page_config(page_title="MCP Web Search Agent", layout="wide")
    st.title("MCP Web Search Agent")
    st.caption("Built using Claude MCP and Python modules")

    # Start MCP server in background if not already running
    if 'server_process' not in st.session_state:
        st.session_state.server_process = start_mcp_server()
        st.session_state.server_running = True

    # Initialize session state variables
    if 'follow_links' not in st.session_state:
        st.session_state.follow_links = True
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar controls
    st.sidebar.title("Settings")
    if st.session_state.server_running:
        st.sidebar.success("MCP Server: Running")
    else:
        st.sidebar.error("MCP Server: Not Running")
        if st.sidebar.button("Start Server"):
            st.session_state.server_process = start_mcp_server()
            st.session_state.server_running = True
            st.rerun()

    # Add search mode toggle in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Options")
    follow_links = st.sidebar.toggle(
        "Follow Links", 
        value=st.session_state.follow_links,
        help="When enabled, follows links from search results to find more information. When disabled, only uses direct search results."
    )
    
    # Update session state if toggle changed
    if follow_links != st.session_state.follow_links:
        st.session_state.follow_links = follow_links
    
    # Display description of current mode
    if st.session_state.follow_links:
        st.sidebar.info("🌐 **Deep Search Mode**: Explores the web by following relevant links from search results to find more comprehensive information.")
    else:
        st.sidebar.info("🔍 **Surface Search Mode**: Only uses information from direct search results without exploring additional links. Faster but may provide less comprehensive answers.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                st.markdown("**Citations:**")
                for citation in message["citations"]:                        
                    st.markdown(f"- [{citation['title']}]({citation['url']})\nCredibility Score: {citation['credibility']}; Similarity Score: {citation['similarity']:.2f}")
            if "visited_links" in message and message["visited_links"]:
                with st.expander("Websites Visited"):
                    for link in message["visited_links"]:
                        st.markdown(f"- {link}")
            if "follow_links_enabled" in message:
                mode_text = "Deep Search" if message["follow_links_enabled"] else "Surface Search"
                st.caption(f"Search Mode: {mode_text}")

    query = st.chat_input("Enter your question")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Searching and generating response...")
            
            try:
                response = run_async_query(query, st.session_state.follow_links)
                answer = response.get("response", "Sorry, I couldn't find an answer.")
                citations = response.get("citations", [])
                follow_links_enabled = response.get("follow_links_enabled", st.session_state.follow_links)

                visited_links = []
                search_results = response.get("search_results", {})
                visited_links = search_results.get("visited_urls", [])
                
                # Show exploration path info if links were followed
                exploration_path = search_results.get("exploration_path", [])
                depth_reached = search_results.get("depth_reached", 0)
                
                message_placeholder.markdown(answer)
                
                mode_text = "Deep Search" if follow_links_enabled else "Surface Search"
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations,
                    "visited_links": visited_links,
                    "follow_links_enabled": follow_links_enabled
                })
                
                if citations:
                    st.markdown("**Citations:**")
                    for citation in citations:
                        st.markdown(f"- [{citation['title']}]({citation['url']})\nCredibility Score: {citation['credibility']}; Similarity Score: {citation['similarity']:.2f}")
                
                if follow_links_enabled and exploration_path:
                    with st.expander(f"Web Exploration Details (Depth: {depth_reached})"):
                        st.markdown("### Exploration Path")
                        for i, step in enumerate(exploration_path):
                            status = "✅" if step.get("success", False) else "❌"
                            title = step.get("title", "Unknown")
                            reason = step.get("reason", "")
                            st.markdown(f"{status} **[{title}]({step.get('url')})**")
                            st.caption(f"Depth: {step.get('depth')}, Reason: {reason}")
                            if not step.get("success", False):
                                st.error(f"Error: {step.get('error', 'Unknown error')}")
                            st.markdown("---")
                
                st.caption(f"Search Mode: {mode_text}")
                
            except Exception as e:
                message_placeholder.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()