import streamlit as st
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
import warnings
import logging
import json
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ActionItem(BaseModel):
    description: str
    priority: str
    compliance_status: str
    timeline: str

class RiskMitigation(BaseModel):
    strategy: str
    stakeholders: List[str]
    steps: List[str]

class ParagraphAnalysis(BaseModel):
    text: str
    actions: List[ActionItem]
    mitigations: List[RiskMitigation]

class DocumentAnalysis(BaseModel):
    context: Dict[str, Any]
    paragraphs: List[ParagraphAnalysis]
    report: str

# --- LLM Factory ---
class LLMFactory:
    _instances: Dict[tuple, Any] = {}

    @classmethod
    def get_llm(cls, api_choice: str, temperature: float, model_name: str = None):
        key = (api_choice, temperature, model_name)
        if key not in cls._instances:
            if api_choice == "groq":
                model = model_name or "mixtral-8x7b-32768"
                cls._instances[key] = ChatGroq(
                    temperature=temperature,
                    model=model,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
            elif api_choice == "google_ai":
                model = model_name or "gemini-2.0-flash"
                cls._instances[key] = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:
                raise ValueError(f"Invalid API choice: {api_choice}")
        return cls._instances[key]

# --- Optimized Base Agent ---
class BaseAgent(Agent):
    def __init__(self, llm, role: str, goal: str, backstory: str):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=10,
            cache=True  # Enable caching for repeated queries
        )

# --- Agent Factory ---
class AgentFactory:
    @staticmethod
    def create_parser(temperature: float, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Regulatory Parser",
            "Parse regulatory documents efficiently",
            "Specialized in document structure analysis"
        )

    @staticmethod
    def create_context(temperature: float, industry: str, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Context Analyzer",
            f"Analyze regulatory context for {industry}",
            "Expert in industry compliance"
        )

    @staticmethod
    def create_action(api_choice: str, temperature: float, industry: str, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm(api_choice, temperature, model_name),
            "Action Extractor",
            f"Extract actionable items for {industry}",
            "Compliance action specialist"
        )

    @staticmethod
    def create_compliance(temperature: float, industry: str, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Compliance Checker",
            f"Verify compliance for {industry}",
            "Regulatory compliance expert"
        )

    @staticmethod
    def create_priority(temperature: float, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Priority Assigner",
            "Assign priority levels",
            "Risk assessment specialist"
        )

    @staticmethod
    def create_mitigation(api_choice: str, temperature: float, industry: str, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm(api_choice, temperature, model_name),
            "Risk Mitigator",
            f"Develop mitigation strategies for {industry}",
            "Risk management expert"
        )

    @staticmethod
    def create_timeline(temperature: float, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Timeline Planner",
            "Plan implementation timelines",
            "Project scheduling expert"
        )

    @staticmethod
    def create_report(temperature: float, model_name: str = None):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature, model_name),
            "Report Generator",
            "Generate structured reports",
            "Documentation specialist"
        )

# --- Core Processing Functions ---
async def parse_document(agent: BaseAgent, content: str) -> List[str]:
    logger.debug("Parsing document content")
    return [p.strip() for p in content.split('\n\n') if p.strip()]

async def process_paragraph(paragraph: str, agents: Dict[str, BaseAgent]) -> ParagraphAnalysis:
    logger.debug(f"Processing paragraph: {paragraph[:50]}...")

    action_task = Task(description=f"Extract actions from: {paragraph[:100]}...", agent=agents["action"])
    compliance_task = Task(description="Check compliance", agent=agents["compliance"])
    priority_task = Task(description="Assign priorities", agent=agents["priority"])
    mitigation_task = Task(description="Suggest mitigations", agent=agents["mitigation"])
    timeline_task = Task(description="Plan timelines", agent=agents["timeline"])

    crew = Crew(agents=list(agents.values()), tasks=[action_task, compliance_task, priority_task, mitigation_task, timeline_task])

    try:
        results = await crew.kickoff()

        # Extract results based on task descriptions
        action_results = next((r for r in results if action_task.description in r), [])
        compliance_results = next((r for r in results if compliance_task.description in r), [])
        priority_results = next((r for r in results if priority_task.description in r), [])
        mitigation_results = next((r for r in results if mitigation_task.description in r), [])
        timeline_results = next((r for r in results if timeline_task.description in r), [])

        actions = []
        for i in range(max(len(action_results), len(priority_results), len(compliance_results), len(timeline_results))):
            actions.append(ActionItem(
                description=str(action_results[i]) if i < len(action_results) else "No action identified",
                priority=str(priority_results[i]) if i < len(priority_results) else "Medium",
                compliance_status=str(compliance_results[i]) if i < len(compliance_results) else "Pending",
                timeline=str(timeline_results[i]) if i < len(timeline_results) else "TBD"
            ))

        mitigations = []
        for m in mitigation_results:
            try:
                if isinstance(m, str):
                    # Attempt to parse the string as JSON
                    mitigation_data = json.loads(m)
                    mitigations.append(RiskMitigation(**mitigation_data))
                elif isinstance(m, dict):
                    mitigations.append(RiskMitigation(**m))
                else:
                    logger.warning(f"Unexpected mitigation format: {type(m)}, skipping.")
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Error processing mitigation: {m}. Error: {e}")

        return ParagraphAnalysis(
            text=paragraph,
            actions=actions,
            mitigations=mitigations
        )

    except Exception as e:
        logger.exception(f"Error processing paragraph: {paragraph[:100]}...")
        return ParagraphAnalysis(text=paragraph, actions=[], mitigations=[])  # Return a default object

# --- Main Processing Function ---
async def process_regulatory_obligation(
    input_type: str,
    input_source: str,
    api_actions: str = "groq",
    api_mitigation: str = "groq",
    temperature: float = 0.5,
    industry: str = "General",
    groq_model_name: str = None,
    google_model_name: str = None
) -> DocumentAnalysis:
    try:
        logger.debug(f"Starting processing with input type: {input_type}, source: {input_source}")

        # Load content
        if input_type == "url":
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(input_source, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        document_content = await resp.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    raise RuntimeError(f"Failed to fetch URL: {e}")
        elif input_type == "file upload":
            document_content = input_source
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if not document_content.strip():
            raise ValueError("Empty document content")

        # Initialize agents
        parser_agent = AgentFactory.create_parser(temperature, groq_model_name)
        context_agent = AgentFactory.create_context(temperature, industry, groq_model_name)
        agents = {
            "action": AgentFactory.create_action(api_actions, temperature, industry, groq_model_name),
            "compliance": AgentFactory.create_compliance(temperature, industry, groq_model_name),
            "priority": AgentFactory.create_priority(temperature, groq_model_name),
            "mitigation": AgentFactory.create_mitigation(api_mitigation, temperature, industry, groq_model_name),
            "timeline": AgentFactory.create_timeline(temperature, groq_model_name),
            "report": AgentFactory.create_report(temperature, groq_model_name)
        }

        # Process document
        paragraphs = await parse_document(parser_agent, document_content)

        # Analyze context - Directly use the agent to avoid unnecessary Crew
        context_task = Task(description="Analyze document context", agent=context_agent)
        context_result = await context_task.execute(context=document_content)

        # Parallel paragraph processing
        paragraph_results = []
        for para in paragraphs:
            result = await process_paragraph(para, agents)
            paragraph_results.append(result)

        # Generate report
        report_agent = agents["report"]  # Get the report agent
        paragraph_context = "\n".join([f"Paragraph {i+1}: {p.text}" for i, p in enumerate(paragraph_results)])  # Joining paragraph contexts
        report_task = Task(
            description="Generate comprehensive report",
            agent=report_agent,
            context=f"Overall context: {context_result}\nParagraph summaries: {paragraph_context}"
        )
        report_result = await report_task.execute(context=f"Overall context: {context_result}\nParagraph summaries: {paragraph_context}")

        logger.debug("Document processed successfully")
        return DocumentAnalysis(
            context={"summary": str(context_result)},
            paragraphs=paragraph_results,
            report=str(report_result)
        )

    except Exception as e:
        logger.exception(f"Processing failed: {str(e)}")
        raise RuntimeError(f"Processing failed: {str(e)}")

# --- Streamlit App ---
async def main_streamlit():
    st.title("Regulatory Obligation Processor")

    input_type = st.selectbox("Input Type", ["file upload", "url"])
    input_source = ""

    if input_type == "file upload":
        uploaded_file = st.file_uploader("Upload a file")
        if uploaded_file is not None:
            input_source = uploaded_file.read().decode("utf-8")  # Read as string
    elif input_type == "url":
        input_source = st.text_input("Enter URL")

    api_actions = st.selectbox("API for Actions", ["groq", "google_ai"])
    api_mitigation = st.selectbox("API for Mitigation", ["groq", "google_ai"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
    industry = st.text_input("Industry", "General")

    if st.button("Process"):
        if not input_source:
            st.error("Please provide input source.")
        else:
            try:
                # Await the coroutine!
                with st.spinner("Processing..."):
                    result = await process_regulatory_obligation(
                        input_type=input_type,
                        input_source=input_source,
                        api_actions=api_actions,
                        api_mitigation=api_mitigation,
                        temperature=temperature,
                        industry=industry
                    )
                st.success("Processing complete!")

                # Display Results (handle potential JSON errors)
                try:
                    st.json(result.model_dump())  # Streamlit can directly display JSON
                except Exception as e:
                    st.error(f"Error displaying results as JSON: {e}")
                    st.write(result)  # Fallback to displaying the object

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                st.stop()  # Stop streamlit

# Run the Streamlit app
if __name__ == "__main__":
    asyncio.run(main_streamlit())
