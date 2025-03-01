import os
import asyncio
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="langchain.agents.json_chat.base")

# --- Pydantic Models for Validation ---
# Define Pydantic models
class ActionItem(BaseModel):
    description: str = Field(..., description="Specific action required")
    priority: str = Field(..., description="High/Medium/Low priority level")
    compliance_status: str = Field(..., description="Compliance status")
    timeline: str = Field(..., description="Implementation timeline")

class RiskMitigation(BaseModel):
    strategy: str = Field(..., description="Mitigation strategy")
    stakeholders: List[str] = Field(..., description="Responsible parties")
    steps: List[str] = Field(..., description="Implementation steps")

class ParagraphAnalysis(BaseModel):
    text: str = Field(..., description="Original paragraph text")
    actions: List[ActionItem] = Field(..., description="Identified actions")
    mitigations: List[RiskMitigation] = Field(..., description="Risk mitigations")

class DocumentAnalysis(BaseModel):
    context: Dict[str, Any] = Field(..., description="Document context analysis")
    paragraphs: List[ParagraphAnalysis] = Field(..., description="Paragraph analyses")
    report: str = Field(..., description="Executive summary report")
    

class LLMInitializationError(Exception):
    pass

def get_google_llm(temperature: float) -> ChatGoogleGenerativeAI:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        raise LLMInitializationError(f"Google AI initialization failed: {str(e)}")
    finally:
        loop.close()

def get_llm(api_choice: str, temperature: float) -> ChatGroq | ChatGoogleGenerativeAI:
    try:
        if api_choice == "groq":
            return ChatGroq(
                temperature=temperature,
                model="mixtral-8x7b-32768",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif api_choice == "google_ai":
            return get_google_llm(temperature)
        raise ValueError("Invalid API choice")
    except Exception as e:
        raise LLMInitializationError(f"LLM initialization failed: {str(e)}")

# --- Enhanced Agent Definitions ---
class BaseAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            verbose=False,
            allow_delegation=True,
            **kwargs
        )

class RegulatoryParserAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Regulatory Document Parser',
            goal='Break down regulatory text into paragraphs for analysis.',
            backstory="Expert in parsing legal and regulatory documents.",
            llm=get_llm("groq", temperature)
        )

    def parse_document(self, document_content: str) -> List[str]:
        try:
            return [p.strip() for p in document_content.split('\n\n') if p.strip()]
        except Exception as e:
            raise ValueError(f"Document parsing failed: {str(e)}")

class ContextAnalyzerAgent(BaseAgent):
    def __init__(self, temperature: float, industry: str):
        super().__init__(
            role='Context Analyzer',
            goal=f'Analyze document context for {industry} industry',
            backstory="Expert in regulatory contexts and industry requirements.",
            llm=get_llm("groq", temperature)
        )

class ActionItemAgent(BaseAgent):
    def __init__(self, api_choice: str, temperature: float, industry: str):
        super().__init__(
            role='Action Item Extractor',
            goal=f'Identify actionable items for {industry} industry',
            backstory="Experienced compliance officer.",
            llm=get_llm(api_choice, temperature)
        )

class ComplianceCheckerAgent(BaseAgent):
    def __init__(self, temperature: float, industry: str):
        super().__init__(
            role='Compliance Checker',
            goal=f'Check compliance for {industry} industry',
            backstory="Expert in regulatory compliance frameworks.",
            llm=get_llm("groq", temperature)
        )

class PriorityAssignerAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Priority Assigner',
            goal='Assign priority levels to action items.',
            backstory="Experienced risk analyst.",
            llm=get_llm("groq", temperature)
        )

class RiskMitigationAgent(BaseAgent):
    def __init__(self, api_choice: str, temperature: float, industry: str):
        super().__init__(
            role='Risk Mitigation Strategist',
            goal=f'Suggest risk mitigations for {industry} industry',
            backstory="Expert in risk management.",
            llm=get_llm(api_choice, temperature)
        )

class TimelinePlannerAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Timeline Planner',
            goal='Suggest timelines for implementing actions.',
            backstory="Expert in project planning and timelines.",
            llm=get_llm("groq", temperature)
        )

class ReportGeneratorAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Report Generator',
            goal='Compile all outputs into a structured report.',
            backstory="Expert in creating professional reports.",
            llm=get_llm("groq", temperature)
        )

# --- Task Definitions ---
def create_tasks(parser_agent, context_agent, action_agent, compliance_agent, priority_agent, mitigation_agent, timeline_agent, report_agent, document_content):
    """Create tasks for the crew with proper dependencies."""
    paragraphs = parser_agent.parse_document(document_content)
    tasks = []

    # Context Analysis Task
    context_task = Task(
        description="Analyze the overall context of the document.",
        agent=context_agent,
        expected_output="Document context including industry, regulatory body, and jurisdiction."
    )
    tasks.append(context_task)

    for idx, paragraph in enumerate(paragraphs):
        # Action item extraction task
        action_task = Task(
            description=f"Extract action items from paragraph {idx+1}",
            agent=action_agent,
            context=[context_task],
            expected_output="List of actionable items from the regulatory paragraph."
        )
        tasks.append(action_task)

        # Compliance check task
        compliance_task = Task(
            description=f"Check compliance for paragraph {idx+1} actions",
            agent=compliance_agent,
            context=[action_task],
            expected_output="Compliance status for each action item."
        )
        tasks.append(compliance_task)

        # Priority assignment task
        priority_task = Task(
            description=f"Assign priority to paragraph {idx+1} actions",
            agent=priority_agent,
            context=[action_task],
            expected_output="Priority levels for each action item."
        )
        tasks.append(priority_task)

        # Risk mitigation task
        mitigation_task = Task(
            description=f"Suggest mitigations for paragraph {idx+1} actions",
            agent=mitigation_agent,
            context=[action_task],
            expected_output="Risk mitigation strategies for the identified action items."
        )
        tasks.append(mitigation_task)

        # Timeline planning task
        timeline_task = Task(
            description=f"Plan timelines for paragraph {idx+1} actions",
            agent=timeline_agent,
            context=[action_task, mitigation_task],
            expected_output="Timelines for implementing actions and mitigations."
        )
        tasks.append(timeline_task)

    # Report generation task
    report_task = Task(
        description="Generate a final report.",
        agent=report_agent,
        context=tasks,
        expected_output="Structured, human-readable report."
    )
    tasks.append(report_task)

    return tasks

# --- Main Processing Function ---
def process_regulatory_obligation(
    input_type: str,
    input_source: str,
    api_actions: str = "groq",
    api_mitigation: str = "groq",
    temperature: float = 0.5,
    industry: str = "General"
) -> DocumentAnalysis:
    try:
        # Initialize agents
        parser_agent = RegulatoryParserAgent(temperature)
        context_agent = ContextAnalyzerAgent(temperature, industry)
        action_agent = ActionItemAgent(api_actions, temperature, industry)
        compliance_agent = ComplianceCheckerAgent(temperature, industry)
        priority_agent = PriorityAssignerAgent(temperature)
        mitigation_agent = RiskMitigationAgent(api_mitigation, temperature, industry)
        timeline_agent = TimelinePlannerAgent(temperature)
        report_agent = ReportGeneratorAgent(temperature)

        # Create crew
        crew = Crew(
            agents=[parser_agent, context_agent, action_agent, compliance_agent, priority_agent, mitigation_agent, timeline_agent, report_agent],
            tasks=[],
            verbose=False
        )

        # Load document content
        document_content = ""
        if input_type == "url":
            response = requests.get(input_source, timeout=10)
            response.raise_for_status()
            document_content = response.text
        elif input_type == "file upload":
            document_content = input_source
        else:
            raise ValueError("Invalid input type")

        if not document_content.strip():
            raise ValueError("Empty document content")

        # Create and execute tasks
        crew.tasks = create_tasks(parser_agent, context_agent, action_agent, compliance_agent, priority_agent, mitigation_agent, timeline_agent, report_agent, document_content)
        results = crew.kickoff()

        # Structure and validate output
        return DocumentAnalysis(
            context=results[0],
            paragraphs=[
                ParagraphAnalysis(
                    text=para,
                    actions=results[idx*5 + 1],
                    mitigations=results[idx*5 + 4]
                ) for idx, para in enumerate(parser_agent.parse_document(document_content))
            ],
            report=results[-1]
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error: {str(e)}")
    except LLMInitializationError as e:
        raise RuntimeError(f"AI service error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")
pass
