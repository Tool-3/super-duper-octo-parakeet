import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import ScrapeWebsiteTool, SerperApiTool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration --- 
# IMPORTANT: Set your API keys as environment variables
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
# os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
# Make sure to install necessary libraries: pip install streamlit crewai crewai-tools langchain-google-genai duckduckgo-search

# --- LLM Setup --- 
# Ensure the GOOGLE_API_KEY environment variable is set
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = None
if google_api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", # Or use "gemini-1.5-pro-latest" if available and preferred
        verbose=True,
        temperature=0.1,
        google_api_key=google_api_key
    )
else:
    st.warning("GOOGLE_API_KEY not found in environment variables. LLM functionality will be limited or unavailable.")

# --- Tool Setup --- 
# Tool for searching the web for RBI notifications/circulars
search_tool = SerperApiTool()

# Tool for scraping content from a specific URL found by the search tool
scrape_tool = ScrapeWebsiteTool()

# --- Agent Definitions --- 
st.title("RBI Regulatory Analysis Crew (using Gemini)")

st.info("**Note:** This is a conceptual example. Accessing and interpreting RBI data accurately requires robust tools and potentially specific scraping logic tailored to rbi.org.in.")

# Placeholder for user input, e.g., specific topics or date ranges
query = st.text_input("Enter search query for RBI documents (e.g., 'RBI Master Directions KYC 2023')", "latest RBI master circulars")

# Agent 1: Researcher
# Finds relevant RBI documents online.
rbi_researcher = Agent(
  role='RBI Document Researcher',
  goal=f'Find relevant notifications, master circulars, and master directions from the RBI website based on the query: {query}',
  backstory=("""
    You are an expert researcher specializing in Indian financial regulations. 
    Your task is to use search tools to find the most relevant and up-to-date 
    official documents published by the Reserve Bank of India (RBI) related to the user's query.
    Focus on finding links to official RBI pages or documents (.pdf, .html).
    """),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool, scrape_tool],
  llm=llm # Specify the Gemini model
)

# Agent 2: Regulatory Analyst
# Extracts obligations from the documents.
regulatory_analyst = Agent(
  role='Regulatory Compliance Analyst',
  goal='Analyze the provided RBI documents and extract key regulatory obligations and requirements for financial institutions.',
  backstory=("""
    You are a meticulous analyst with deep knowledge of banking regulations. 
    Your job is to read through the RBI documents identified by the researcher, 
    understand the context, and clearly list the specific actions, rules, or standards 
    that financial institutions must comply with.
    """),
  verbose=True,
  allow_delegation=False,
  llm=llm # Specify the Gemini model
)

# Agent 3: Risk Assessor
# Identifies risks and suggests mitigation strategies.
risk_assessor = Agent(
  role='Financial Risk Management Expert',
  goal='Identify potential risks associated with the extracted regulatory obligations and suggest practical mitigation strategies.',
  backstory=("""
    You are an experienced risk management professional. Based on the regulatory 
    obligations identified by the analyst, you anticipate potential risks (operational, 
    compliance, financial, reputational) for financial institutions and propose 
    concrete steps and controls to mitigate these risks effectively.
    """),
  verbose=True,
  allow_delegation=False,
  llm=llm # Specify the Gemini model
)

# --- Task Definitions --- 

# Task 1: Find RBI Documents
find_docs_task = Task(
  description=("""
    Search the web for official RBI notifications, master circulars, or master directions 
    related to '{query}'. Identify the most relevant URLs or document sources. 
    If possible, scrape the content of the primary sources found. Provide the findings as context for the next agent.
    Focus on rbi.org.in domain if possible.
    """.format(query=query)),
  expected_output='A list of relevant RBI document URLs and, if possible, their scraped text content or summaries.',
  agent=rbi_researcher
)

# Task 2: Extract Obligations
extract_obligations_task = Task(
  description=("""
    Review the documents and content provided by the RBI Document Researcher. 
    Identify and list the specific regulatory obligations, requirements, and compliance mandates mentioned. 
    Be precise and clear.
    """),
  expected_output='A structured list of key regulatory obligations extracted from the RBI documents.',
  agent=regulatory_analyst,
  context=[find_docs_task] # Depends on the output of the first task
)

# Task 3: Assess Risks and Mitigation
assess_risk_task = Task(
  description=("""
    Analyze the list of regulatory obligations provided by the Regulatory Compliance Analyst. 
    For each significant obligation, identify potential risks for financial institutions 
    and propose corresponding mitigation strategies or controls.
    """),
  expected_output='A report outlining potential risks linked to the identified obligations and suggested mitigation actions.',
  agent=risk_assessor,
  context=[extract_obligations_task] # Depends on the output of the second task
)

# --- Crew Definition --- 

rbi_crew = Crew(
  agents=[rbi_researcher, regulatory_analyst, risk_assessor],
  tasks=[find_docs_task, extract_obligations_task, assess_risk_task],
  process=Process.sequential, # Tasks will be executed sequentially
  verbose=2 # Shows agent reasoning and actions
)

# --- Streamlit Execution --- 

if st.button("Start Analysis"):
    # Check for necessary API keys
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found! Please set the GOOGLE_API_KEY environment variable.")
    elif not serper_api_key:
        st.error("SERPER_API_KEY not found! Please set the SERPER_API_KEY environment variable for the search tool.")
    elif llm is None: # Extra check in case llm initialization failed silently
         st.error("LLM could not be initialized. Check API key and configuration.")
    else:
        with st.spinner("Crew is analyzing RBI regulations using Gemini... Please wait."):
            try:
                result = rbi_crew.kickoff()
                
                st.subheader("Analysis Results:")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"An error occurred during crew execution: {e}")
                st.exception(e) # Provides full traceback
