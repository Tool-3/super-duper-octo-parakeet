import os
import asyncio
from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

# --- Helper Function to Initialize Google AI LLM ---
def get_google_llm(temperature):
    """Initialize Google AI LLM with proper async event loop handling."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    finally:
        loop.close()

# --- Configure LLMs ---
def get_llm(api_choice, temperature):
    if api_choice == "groq":
        return ChatGroq(
            temperature=temperature,
            model="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    elif api_choice == "google_ai":
        return get_google_llm(temperature)
    else:
        raise ValueError("Invalid API choice.")

# --- Agent Definitions ---
class RegulatoryParserAgent(Agent):
    def __init__(self, temperature):
        super().__init__(
            role='Regulatory Document Parser',
            goal='Break down regulatory text into paragraphs for analysis.',
            backstory="Expert in parsing legal and regulatory documents.",
            verbose=False,
            allow_delegation=True,
            llm=get_llm("groq", temperature)
        )

    def parse_document(self, document_content):
        """Parse document content into paragraphs."""
        return [p.strip() for p in document_content.split('\n\n') if p.strip()]

class ContextAnalyzerAgent(Agent):
    def __init__(self, temperature, industry):
        super().__init__(
            role='Context Analyzer',
            goal=f'Analyze the overall context of the document for the {industry} industry.',
            backstory="Expert in understanding regulatory contexts and industry-specific requirements.",
            verbose=False,
            llm=get_llm("groq", temperature)
        )

class ActionItemAgent(Agent):
    def __init__(self, api_choice, temperature, industry):
        llm = get_llm(api_choice, temperature)
        super().__init__(
            role='Action Item Extractor',
            goal=f'Identify actionable items from regulatory text for the {industry} industry.',
            backstory="Experienced compliance officer.",
            verbose=False,
            allow_delegation=True,
            llm=llm
        )

class ComplianceCheckerAgent(Agent):
    def __init__(self, temperature, industry):
        super().__init__(
            role='Compliance Checker',
            goal=f'Check action items against compliance frameworks for the {industry} industry.',
            backstory="Expert in regulatory compliance and frameworks.",
            verbose=False,
            llm=get_llm("groq", temperature)
        )

class PriorityAssignerAgent(Agent):
    def __init__(self, temperature):
        super().__init__(
            role='Priority Assigner',
            goal='Assign priority levels to action items based on risk and impact.',
            backstory="Experienced risk analyst.",
            verbose=False,
            llm=get_llm("groq", temperature)
        )

class RiskMitigationAgent(Agent):
    def __init__(self, api_choice, temperature, industry):
        llm = get_llm(api_choice, temperature)
        super().__init__(
            role='Risk Mitigation Strategist',
            goal=f'Suggest risk mitigation strategies for the {industry} industry.',
            backstory="Expert in risk management.",
            verbose=False,
            llm=llm
        )

class TimelinePlannerAgent(Agent):
    def __init__(self, temperature):
        super().__init__(
            role='Timeline Planner',
            goal='Suggest timelines for implementing action items and mitigations.',
            backstory="Expert in project planning and timelines.",
            verbose=False,
            llm=get_llm("groq", temperature)
        )

class ReportGeneratorAgent(Agent):
    def __init__(self, temperature):
        super().__init__(
            role='Report Generator',
            goal='Compile all outputs into a structured, human-readable report.',
            backstory="Expert in creating professional reports.",
            verbose=False,
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
def process_regulatory_obligation(input_type, input_source, api_actions="groq", api_mitigation="groq", temperature=0.5, industry="General"):
    """Process regulatory obligation from URL or file."""
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
        try:
            response = requests.get(input_source)
            response.raise_for_status()
            document_content = response.text
        except Exception as e:
            return f"Error loading URL: {str(e)}"
    elif input_type == "file upload":
        document_content = input_source
    else:
        return "Invalid input type. Use 'url' or 'file upload'."

    if not document_content.strip():
        return "No valid document content found."

    # Create and execute tasks
    crew.tasks = create_tasks(parser_agent, context_agent, action_agent, compliance_agent, priority_agent, mitigation_agent, timeline_agent, report_agent, document_content)
    results = crew.kickoff()

    # Structure results
    output = {
        "context": results[0],  # Context analysis
        "paragraphs": []
    }
    paragraphs = parser_agent.parse_document(document_content)
    for idx, para in enumerate(paragraphs):
        output["paragraphs"].append({
            "text": para,
            "actions": results[idx*5 + 1],  # Action items
            "compliance": results[idx*5 + 2],  # Compliance status
            "priority": results[idx*5 + 3],  # Priority levels
            "mitigations": results[idx*5 + 4],  # Risk mitigations
            "timeline": results[idx*5 + 5]  # Timelines
        })
    output["report"] = results[-1]  # Final report

    return output
