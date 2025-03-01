import os
import asyncio
import aiohttp
from typing import List, Dict, Any
from pydantic import BaseModel
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
    def get_llm(cls, api_choice: str, temperature: float):
        key = (api_choice, temperature)
        if key not in cls._instances:
            if api_choice == "groq":
                cls._instances[key] = ChatGroq(
                    temperature=temperature,
                    model="mixtral-8x7b-32768",
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
            elif api_choice == "google_ai":
                cls._instances[key] = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
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
    def create_parser(temperature: float):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Regulatory Parser",
            "Parse regulatory documents efficiently",
            "Specialized in document structure analysis"
        )

    @staticmethod
    def create_context(temperature: float, industry: str):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Context Analyzer",
            f"Analyze regulatory context for {industry}",
            "Expert in industry compliance"
        )

    @staticmethod
    def create_action(api_choice: str, temperature: float, industry: str):
        return BaseAgent(
            LLMFactory.get_llm(api_choice, temperature),
            "Action Extractor",
            f"Extract actionable items for {industry}",
            "Compliance action specialist"
        )

    @staticmethod
    def create_compliance(temperature: float, industry: str):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Compliance Checker",
            f"Verify compliance for {industry}",
            "Regulatory compliance expert"
        )

    @staticmethod
    def create_priority(temperature: float):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Priority Assigner",
            "Assign priority levels",
            "Risk assessment specialist"
        )

    @staticmethod
    def create_mitigation(api_choice: str, temperature: float, industry: str):
        return BaseAgent(
            LLMFactory.get_llm(api_choice, temperature),
            "Risk Mitigator",
            f"Develop mitigation strategies for {industry}",
            "Risk management expert"
        )

    @staticmethod
    def create_timeline(temperature: float):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Timeline Planner",
            "Plan implementation timelines",
            "Project scheduling expert"
        )

    @staticmethod
    def create_report(temperature: float):
        return BaseAgent(
            LLMFactory.get_llm("groq", temperature),
            "Report Generator",
            "Generate structured reports",
            "Documentation specialist"
        )

# --- Core Processing Functions ---
async def parse_document(agent: BaseAgent, content: str) -> List[str]:
    return [p.strip() for p in content.split('\n\n') if p.strip()]

async def process_paragraph(paragraph: str, agents: Dict[str, BaseAgent]) -> ParagraphAnalysis:
    tasks = [
        Task(description=f"Extract actions from: {paragraph[:100]}...", agent=agents["action"]),
        Task(description="Check compliance", agent=agents["compliance"]),
        Task(description="Assign priorities", agent=agents["priority"]),
        Task(description="Suggest mitigations", agent=agents["mitigation"]),
        Task(description="Plan timelines", agent=agents["timeline"])
    ]
    
    crew = Crew(agents=list(agents.values()), tasks=tasks)
    results = await asyncio.to_thread(crew.kickoff)

    actions = []
    for i in range(len(results[0] or [])):
        actions.append(ActionItem(
            description=str(results[0][i]) if results[0] else "No action identified",
            priority=str(results[2][i]) if results[2] else "Medium",
            compliance_status=str(results[1][i]) if results[1] else "Pending",
            timeline=str(results[4][i]) if results[4] else "TBD"
        ))

    mitigations = [RiskMitigation(**m) for m in (results[3] or []) if isinstance(m, dict)]
    
    return ParagraphAnalysis(
        text=paragraph,
        actions=actions,
        mitigations=mitigations
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
    try:
        # Load content
        if input_type == "url":
            async with aiohttp.ClientSession() as session:
                async with session.get(input_source, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    document_content = await resp.text()
        elif input_type == "file upload":
            document_content = input_source
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if not document_content.strip():
            raise ValueError("Empty document content")

        # Initialize agents
        parser_agent = AgentFactory.create_parser(temperature)
        context_agent = AgentFactory.create_context(temperature, industry)
        agents = {
            "action": AgentFactory.create_action(api_actions, temperature, industry),
            "compliance": AgentFactory.create_compliance(temperature, industry),
            "priority": AgentFactory.create_priority(temperature),
            "mitigation": AgentFactory.create_mitigation(api_mitigation, temperature, industry),
            "timeline": AgentFactory.create_timeline(temperature),
            "report": AgentFactory.create_report(temperature)
        }

        # Process document
        paragraphs = await parse_document(parser_agent, document_content)
        
        context_task = Task(description="Analyze document context", agent=context_agent)
        context_crew = Crew(agents=[context_agent], tasks=[context_task])
        context_result = await asyncio.to_thread(context_crew.kickoff)

        # Parallel paragraph processing
        async with ThreadPoolExecutor(max_workers=4) as executor:
            paragraph_tasks = [process_paragraph(para, agents) for para in paragraphs]
            paragraph_results = await asyncio.gather(*paragraph_tasks)

        # Generate report
        report_task = Task(
            description="Generate comprehensive report",
            agent=agents["report"],
            context=[context_task] + [Task(description=f"Paragraph {i}", agent=agents["report"]) 
                                    for i in range(len(paragraphs))]
        )
        report_crew = Crew(agents=[agents["report"]], tasks=[report_task])
        report_result = await asyncio.to_thread(report_crew.kickoff)

        return DocumentAnalysis(
            context=context_result if isinstance(context_result, dict) else {"summary": str(context_result)},
            paragraphs=paragraph_results,
            report=str(report_result)
        )

    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

# --- Entry Point ---
if __name__ == "__main__":
    async def main():
        result = await process_regulatory_obligation(
            input_type="file upload",
            input_source="Sample regulatory text here...\n\nSecond paragraph here..."
        )
        print(result.model_dump_json(indent=2))

    asyncio.run(main())
