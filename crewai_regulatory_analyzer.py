import os
import asyncio
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="langchain.agents.json_chat.base")

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

# --- LLM Initialization ---
class LLMInitializationError(Exception):
    pass

@lru_cache(maxsize=2)
def get_llm(api_choice: str, temperature: float) -> ChatGroq | ChatGoogleGenerativeAI:
    if api_choice == "groq":
        return ChatGroq(
            temperature=temperature,
            model="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    elif api_choice == "google_ai":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    raise ValueError("Invalid API choice")

# --- Optimized Agent Base Class ---
class BaseAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            verbose=False,
            allow_delegation=False,
            cache=True,
            **kwargs
        )

# --- Specialized Agents ---
class RegulatoryParserAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Regulatory Document Parser',
            goal='Efficiently parse regulatory text',
            backstory="Optimized document parser",
            llm=get_llm("groq", temperature)
        )

    @lru_cache(maxsize=100)
    def parse_document(self, document_content: str) -> tuple:
        return tuple(p.strip() for p in document_content.split('\n\n') if p.strip())

class ContextAnalyzerAgent(BaseAgent):
    def __init__(self, temperature: float, industry: str):
        super().__init__(
            role='Context Analyzer',
            goal=f'Quick context analysis for {industry}',
            backstory="Efficient context specialist",
            llm=get_llm("groq", temperature)
        )

class CombinedAnalysisAgent(BaseAgent):
    def __init__(self, api_choice: str, temperature: float, industry: str):
        super().__init__(
            role='Combined Analyzer',
            goal=f'Comprehensive analysis for {industry}',
            backstory="Multi-task compliance specialist",
            llm=get_llm(api_choice, temperature)
        )

class ReportGeneratorAgent(BaseAgent):
    def __init__(self, temperature: float):
        super().__init__(
            role='Report Generator',
            goal='Generate concise reports',
            backstory="Efficient report compiler",
            llm=get_llm("groq", temperature)
        )

# --- Optimized Task Processing ---
async def analyze_paragraph(
    paragraph: str,
    context: Dict[str, Any],
    combined_agent: CombinedAnalysisAgent,
    semaphore: asyncio.Semaphore
) -> ParagraphAnalysis:
    async with semaphore:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: combined_agent.execute_task(
                f"Analyze paragraph: '{paragraph}' with context: {context}. "
                "Return action items with priority, compliance status, timeline, and mitigations."
            )
        )
        return ParagraphAnalysis.parse_obj(result)

async def process_document(
    document_content: str,
    parser_agent: RegulatoryParserAgent,
    context_agent: ContextAnalyzerAgent,
    combined_agent: CombinedAnalysisAgent,
    report_agent: ReportGeneratorAgent
) -> DocumentAnalysis:
    paragraphs = parser_agent.parse_document(document_content)
    
    context = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: context_agent.execute_task("Analyze document context")
    )
    
    semaphore = asyncio.Semaphore(5)
    paragraph_tasks = [
        analyze_paragraph(para, context, combined_agent, semaphore)
        for para in paragraphs
    ]
    paragraph_analyses = await asyncio.gather(*paragraph_tasks)
    
    report = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: report_agent.execute_task(
            f"Generate report from context: {context} and analyses: {paragraph_analyses}"
        )
    )
    
    return DocumentAnalysis(
        context=context,
        paragraphs=paragraph_analyses,
        report=report
    )

# --- Main Function (Fixed Signature) ---
async def process_regulatory_obligation(
    input_type: str,
    input_source: str,
    api_analysis: str = "groq",
    temperature: float = 0.5,
    industry: str = "General"
) -> DocumentAnalysis:
    try:
        parser_agent = RegulatoryParserAgent(temperature)
        context_agent = ContextAnalyzerAgent(temperature, industry)
        combined_agent = CombinedAnalysisAgent(api_analysis, temperature, industry)
        report_agent = ReportGeneratorAgent(temperature)

        if input_type == "url":
            async with requests.get(input_source) as response:  # Removed timeout parameter for simplicity
                response.raise_for_status()
                document_content = await response.text()
        elif input_type == "file upload":
            document_content = input_source
        else:
            raise ValueError("Invalid input type")

        if not document_content.strip():
            raise ValueError("Empty document content")

        return await process_document(
            document_content,
            parser_agent,
            context_agent,
            combined_agent,
            report_agent
        )

    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

# --- Entry Point (Fixed Call) ---
if __name__ == "__main__":
    # Use keyword arguments for clarity and to match function signature
    result = asyncio.run(
        process_regulatory_obligation(
            input_type="file upload",
            input_source="Sample regulatory text here...",
            api_analysis="groq",  # Optional, matches default
            temperature=0.5,      # Optional, matches default
            industry="General"    # Optional, matches default
        )
    )
    print(result.json())
