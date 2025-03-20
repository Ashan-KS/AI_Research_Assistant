import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools import duck_search_tool, wiki_tool, save_tool
import json

load_dotenv()

# Class to set the structure of the responses
class ResponseStructure(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the model and response parser
llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResponseStructure)

# Creating the prompt
prompt = ChatPromptTemplate([
    ("system", """
    You are a research assistant chatbot. Use both Wikipedia and DuckDuckGo search tools as needed.
    If a query requires broad or real-time information, prefer DuckDuckGo. For factual knowledge, use Wikipedia.
    Format responses using:
    {format_instructions}
    """),
    ("placeholder", "{chat_history}"),  # Maintains conversation history
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

# Providing tools for the agent
tools = [duck_search_tool, save_tool, wiki_tool]

# Creating the agent
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Executing the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.title("AI Research Assistant")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history only if there are messages
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
query = st.chat_input("Ask me anything...")

# Ensure raw_response is always defined
raw_response = {}

if query:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Invoke AI response
    raw_response = agent_executor.invoke({"query": query, "chat_history": st.session_state.chat_history})

try:
    raw_output = raw_response.get("output", "").strip()

    # Check if output is JSON
    if raw_output.startswith("{") and raw_output.endswith("}"):
        try:
            json_data = json.loads(raw_output)  # Attempt to parse JSON
            structured_response = ResponseStructure(**json_data)

            response_text = (
                f"**Topic:** {structured_response.topic}\n\n"
                f"**Summary:** {structured_response.summary}\n\n"
                f"**Sources:** {', '.join(structured_response.sources)}\n\n"
                f"**Tools Used:** {', '.join(structured_response.tools_used)}"
            )
        except json.JSONDecodeError:
            response_text = f"**Response:** {raw_output}"  # Handle case where JSON is malformed
    else:
        # If it's not JSON, just return the raw text output
        response_text = f"**Response:** {raw_output}"

except Exception as e:
    response_text = f"⚠️ Error parsing response: {e}\n\nRaw Response: {raw_response}"

# ✅ FIX: Only append and display the response if there's valid content
if query and response_text.strip():
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)