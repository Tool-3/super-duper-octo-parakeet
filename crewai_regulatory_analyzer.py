import os
import asyncio
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="langchain.agents.json_chat.base")

# --- Optimized Pydantic Models ---
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

# --- LLM Initialization ---
class LLMFactory:
    _instances = {}

    @staticmethod
    def get_llm(api_choice: str, temperature: float) -> ChatGroq | ChatGoogleGenerativeAI:
        key = (api_choice, temperature)
        if key not in LLMFactory._instances:
            if api_choice == "groq":
                LLMFactory._instances[key] = ChatGroq(
                    temperature=temperature,
                    model="mixtral-8x7b-32768",
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
            elif api_choice == "google_ai":
                LLMFactory._instances[key] = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=temperature,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:
                raise ValueError("Invalid API choice")
        return LLMFactory._instances[key]

# --- Optimized Agent Base Class ---
class BaseAgent(Agent):
    def __init__(self, llm, role: str, goal: str, backstory: str):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            verbose=False,
            allow_delegation=False,  # Reduced delegation for efficiency
            max_iter=10,  # Limit iterations to prevent runaway processes
        )

# --- Agent Definitions with Memoization ---
class AgentFactory:
    @staticmethod
    def create_parser_agent(temperature: float):
        return BaseAgent(
            llm=LLMFactory.get_llm("groq", temperature),
            role="Regulatory Document Parser",
            goal="Efficiently parse regulatory documents",
            backstory="Optimized document parser"
        )

    @staticmethod
    def create_context_agent(temperature: float, industry: str):
        return BaseAgent(
            llm=LLMFactory.get_llm("groq", temperature),
            role="Context Analyzer",
            goal=f"Analyze document context for {industry}",
            backstory="Industry context specialist"
        )

    # Add other agent creators similarly...

# --- Optimized Task Execution ---
async def process_paragraph(paragraph: str, agents: Dict[str, BaseAgent]) -> ParagraphAnalysis:
    action_task = Task(description=f"Extract actions from: {paragraph}", agent=agents["action"])
    compliance_task = Task(description="Check compliance", agent=agents["compliance"], context=[action_task])
    priority_task = Task(description="Assign priorities", agent=agents["priority"], context=[action_task])
    mitigation_task = Task(description="Suggest mitigations", agent=agents["mitigation"], context=[action_task])
    timeline_task = Task(description="Plan timelines", agent=agents["timeline"], context=[action_task, mitigation_task])

    crew = Crew(agents=list(agents.values()), tasks=[action_task, compliance_task, priority_task, mitigation_task, timeline_task])
    results = await asyncio.to_thread(crew.kickoff)

    return ParagraphAnalysis(
        text=paragraph,
        actions=[ActionItem(**{**results[0][i], **{"compliance_status": results[1][i], "timeline": results[4][i]}})
                for i in range(len(results[0]))],
        mitigations=results[3]
    )

# --- Main Processing Function ---
async def process_regulatory_obligation(
    input_type: str,
    input_source: str,
    api_actions: str = "groq",
    api_mitigation: str = "groq",
    temperature: float = 0.5,
    industry: str = "General"
) -> DocumentAnalysis:
    # Load document content
    if input_type == "url":
        async with requests.get(input_source, timeout=10) as response:
            document_content = await response.text()
    elif input_type == "file upload":
        document_content = input_source
    else:
        raise ValueError("Invalid input type")

    if not document_content.strip():
        raise ValueError("Empty document content")

    # Initialize agents
    parser_agent = AgentFactory.create_parser_agent(temperature)
    context_agent = AgentFactory.create_context_agent(temperature, industry)
    agents = {
        "action": BaseAgent(LLMFactory.get_llm(api_actions, temperature), "Action Extractor", f"Extract actions for {industry}", "Compliance specialist"),
        "compliance": BaseAgent(LLMFactory.get_llm("groq", temperature), "Compliance Checker", f"Check compliance for {industry}", "Compliance expert"),
        "priority": BaseAgent(LLMFactory.get_llm("groq", temperature), "Priority Assigner", "Assign priorities", "Risk analyst"),
        "mitigation": BaseAgent(LLMFactory.get_llm(api_mitigation, temperature), "Risk Mitigator", f"Mitigate risks for {industry}", "Risk manager"),
        "timeline": BaseAgent(LLMFactory.get_llm("groq", temperature), "Timeline Planner", "Plan timelines", "Project planner"),
        "report": BaseAgent(LLMFactory.get_llm("groq", temperature), "Report Generator", "Generate reports", "Report specialist")
    }

    # Process document
    paragraphs = parser_agent.parse_document(document_content)
    context_task = Task(description="Analyze document context", agent=context_agent)
    context_result = await asyncio.to_thread(Crew(agents=[context_agent], tasks=[context_task]).kickoff)

    # Parallel paragraph processing
    with ThreadPoolExecutor() as executor:
        paragraph_tasks = [process_paragraph(para, agents) for para in paragraphs]
        paragraph_results = await asyncio.gather(*paragraph_tasks)

    # Generate report
    report_task = Task(description="Generate final report", agent=agents["report"], context=[context_task] + paragraph_tasks)
    report_result = await asyncio.to_thread(Crew(agents=[agents["report"]], tasks=[report_task]).kickoff)

    return DocumentAnalysis(
        context=context_result,
        paragraphs=paragraph_results,
        report=report_result
    )

# Example usage
if __name__ == "__main__":
    result = asyncio.run(process_regulatory_obligation("file upload", "Sample regulatory text here..."))
