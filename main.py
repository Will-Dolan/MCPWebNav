import streamlit as st
import os
from dotenv import load_dotenv
from backend_api import query_web_agent
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load API key securely
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

def main():
    st.set_page_config(page_title="MCP Web Search Agent", layout="wide")
    st.title("MCP Web Search Agent")
    st.caption("Built using Claude MCP and Python modules")

    query = st.text_input("Enter your question", placeholder="e.g. What is quantum computing?")

    if st.button("Search") and query:
        with st.spinner("Searching and generating response..."):
            try:
                response = query_web_agent(query, api_key)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()

            st.subheader("Answer")
            st.write(response["answer"])

            st.subheader("Citations")
            for citation in response.get("citations", []):
                st.markdown(
                    f"- [{citation['source']}]({citation['url']}) — *{citation.get('snippet', '')}*"
                )

            st.subheader("Websites Visited")
            for link in response.get("visited_links", []):
                st.markdown(f"- {link}")

            if response.get("related_questions"):
                st.subheader("Related Questions")
                for rq in response["related_questions"]:
                    st.button(rq)


if __name__=='__main__':
    main()