# Common imports
import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import time
import re

# Detect if running on Streamlit Cloud (secrets available)
IS_STREAMLIT_CLOUD = bool(st.secrets)

if IS_STREAMLIT_CLOUD:
    # Load from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    exa_api_key = st.secrets["EXA_API_KEY"]

    # Create a temporary Google service account JSON file
    from google.oauth2 import service_account
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmpfile:
        json.dump(dict(st.secrets["gcp_service_account"]), tmpfile)
        SERVICE_ACCOUNT_PATH = tmpfile.name

else:
    # Load from local .env
    load_dotenv('.env')
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")
    SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Import CrewAI classes
from crewai import Agent, Task, Crew
from crewai_tools import WebsiteSearchTool, EXASearchTool
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from content.driver import download_drive_files, build_rag_tool_from_files

# Google Drive folder
FOLDER_ID = "1B8fvzo_LiLbXDp3Y2vq8yWHOOH_V-BD-"

# Step 1: Download files
downloaded_files = download_drive_files(FOLDER_ID, SERVICE_ACCOUNT_PATH)

# Step 2: Create RAG tool and Exa search tool
rag_tool = build_rag_tool_from_files(downloaded_files)
exa_search_tool = EXASearchTool(api_key=exa_api_key)

# Create a new instance of the WebsiteSearchTool
# Set the base URL of a website, e.g., "https://example.com/", so that the tool can search for sub-pages on that website
#wellness_tool = WebsiteSearchTool("https://isomer-user-content.by.gov.sg/3/c60453c6-3290-42c4-8e0d-97f1b0f87a29/national-mental-health-and-well-being-strategy-report-(final).pdf")

#Agent 1: A friend/listening ear to talk to
agent_companion = Agent(
    role="Listening Companion",
    goal="Be a supportive and understanding friend who listens to workplace struggles. Tailor your response to the user's specific situation, role, and details from their query. Always base your advice and examples on the user's input topic: {topic}. Reference the most relevant resources from RAG tools and avoid generic advice. Use a tone that matches the user's needs (e.g., supportive, empathetic). Only reply if the user's input is relevant to emotional support; otherwise, do not reply.",
    backstory="""You're a thoughtful and empathetic listener, always ready to lend a virtual ear.
    You don’t judge — you just help people feel heard and understood when work gets overwhelming.""",
    #tools=[wellness_tool],
    tools=[rag_tool,exa_search_tool],
    allow_delegation=False,
    verbose=True
)

#Agent 2: Planner
agent_planner = Agent(
    role="Stress Relief Planner",
    goal="Create actionable, short-term and long-term plans or provide perspectives and tips to reduce workplace stress, tailored to the user's specific challenges, role, and context. Use details from the user's query and always base your suggestions on the user's input topic: {topic}. Reference the most relevant resources from RAG tools. Avoid generic suggestions and write in a practical, encouraging style. Only reply if the user's input is relevant to planning or stress relief; otherwise, do not reply.",
    backstory="""You're a productivity and wellness planner who breaks stressors into manageable tasks,
    and suggests realistic and practical routines and positively affect the affected party to help them regain control either physical, emotionally or mentally.""",
    #tools=[wellness_tool],
    tools=[rag_tool,exa_search_tool],
    allow_delegation=False,
    verbose=True
)

#Agent 3: Ally/Mentor
agent_mentor = Agent(
    role="Supportive Mentor",
    goal="Offer guidance on how to help a colleague who may be in distress or struggling mentally, tailored to the mentor's specific situation and the details provided. Only offer this guidance if the affected party is a mentor seeking help. Always base your guidance on the user's input topic: {topic}. Reference the most relevant resources from RAG tools and avoid generic advice. Use a tone that is experienced and supportive. Only reply if the user's input is relevant to mentoring or allyship; otherwise, do not reply.",
    backstory="""You're a extremely experienced workplace mentor who advises others on how to be a good ally in emotionally sensitive situations. 
    You have helped many colleagues, peers or subordinates to overcome their mental wellness challenges.""",
    #tools=[wellness_tool],
    tools=[rag_tool,exa_search_tool],
    allow_delegation=False,
    verbose=True
)

#Agent 4: Guides and Resources
agent_resource = Agent(
    role="Resource Scout",
    goal="Find useful guides, articles, and tips related to workplace wellness, mental health, and stress management, tailored to the user's specific needs and context. Use details from the user's query and always base your recommendations on the user's input topic: {topic}. Reference the most relevant resources from RAG tools. Avoid generic or irrelevant links, and write in a practical, resourceful style. Only reply if the user's input is relevant to resources or guides; otherwise, do not reply.",
    backstory="""You’re a proactive and reliable guide-finder. When someone is struggling or curious about how to improve their mental wellness,
    you find reputable sources and practical guides that can help them to overcome their mental wellness challenges or at least change their perspectives more positively""",
    #tools=[wellness_tool],
    tools=[rag_tool,exa_search_tool],
    allow_delegation=False,
    verbose=True
)

agent_coordinator = Agent(
    role="Crew Coordinator",
    goal="Read the outputs of each specialist agent and produce a unified final message for the user.",
    backstory="You've reviewed the messages from our Listening Companion, Stress Relief Planner, Supportive Mentor, and Resource Scout.",
    allow_delegation=False,
    verbose=True
)
###Tasks

task_companion = Task(
    description="Offer empathetic support and simple advice for someone feeling burnt out or has some mental wellness challenges. Always base your support and advice on the user's input topic: {topic}. Keep your response concise and under 100 words.",
    expected_output="Begin with a few supportive messages acknowledging their feelings and a few gentle coping suggestions to cope with workplace stress. Keep the response under 100 words.",
    agent=agent_companion,
)

task_planner = Task(
    description="Create a step-by-step plan to help someone organise their workweek to reduce stress or their mental wellness challenges. Always base your plan on the user's input topic: {topic}. Keep your response concise and under 100 words.",
    expected_output="Include a practical, clear, easy-to-follow plan with healthy habits, breaks, and boundary-setting ideas. Keep the response under 100 words.",
    agent=agent_planner,
)

task_mentor = Task(
    description="Suggest kind, appropriate ways to support a colleague who may be overwhelmed or has some mental wellness challenges. Always base your suggestions on the user's input topic: {topic}. Keep your response concise and under 100 words.",
    expected_output="Include tips on how to approach, listen, and offer help to someone struggling. Keep the response under 100 words.",
    agent=agent_mentor,
)

task_resource = Task(
    description="Look up trustworthy resources related to managing workplace stress or burnout. Always base your recommendations on the user's input topic: {topic}. Keep your response concise and under 100 words.",
    expected_output="Include not more than 2 useful links or summaries of guides that can help the user take practical steps. Keep the response under 100 words.",
    agent=agent_resource,
)

task_coordinator = Task(
    description="Combine the insights from emotional support, planning steps, mentoring tips, and trusted resources into one final output. Keep the entire response under 300 words.",
    expected_output="A summarised version of a cohesive answer that weaves together emotional empathy, actionable plan, handy tips, and resource links. Structure the answer into clear sections with headings: 'Emotional Support', 'Actionable Plan', 'Mentoring Tips', and 'Recommended Resources'. The answer should be within 300 word count.",
    agent=agent_coordinator,
    context=[task_companion, task_planner, task_mentor, task_resource],
)

crew = Crew(
    agents=[agent_companion, agent_planner, agent_mentor, agent_resource, agent_coordinator],
    tasks=[task_companion, task_planner, task_mentor, task_resource, task_coordinator],
    verbose=False,  # Set verbose to False for faster execution
    concurrent=True  # Enable concurrent execution of agents/tasks
)

def route_to_agent(prompt):
    lower_prompt = prompt.lower()
    agent_keywords = {
        "Companion": ["listen", "support", "feel", "emotion", "burnt out", "overwhelm", "talk", "vent", "stressed", "anxious", "sad", "frustrated"],
        "Planner": ["plan", "organize", "routine", "schedule", "workweek", "stress relief", "time management", "productivity", "breaks", "habits", "structure"],
        "Mentor": ["mentor", "colleague", "ally", "help others", "guidance", "coach", "advice for others", "team", "peer", "subordinate"],
        "Resource": ["resource", "guide", "article", "link", "tips", "reference", "information", "read", "learn", "material"]
    }
    for agent, keywords in agent_keywords.items():
        if any(word in lower_prompt for word in keywords):
            return agent
    # Instead of defaulting to Coordinator, return None if no match
    return None

# Streaming response generator using CrewAI
AGENT_TASK_MAP = {
    "Companion": (agent_companion, task_companion),
    "Planner": (agent_planner, task_planner),
    "Mentor": (agent_mentor, task_mentor),
    "Resource": (agent_resource, task_resource),
    "Coordinator": (agent_coordinator, task_coordinator)
}

def response_generator_from_crewai(user_input):
    import streamlit as st
    history = st.session_state.get("messages", [])
    context_str = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
    ])
    # Try to infer agent from previous response if current input is unclear
    selected = route_to_agent(user_input)
    if selected is None and history:
        # Look for last assistant response and infer agent type from its section heading
        last_response = next((msg['content'] for msg in reversed(history) if msg['role'] == 'assistant'), None)
        if last_response:
            # Simple heading-based inference
            if "Emotional Support" in last_response:
                selected = "Companion"
            elif "Actionable Plan" in last_response:
                selected = "Planner"
            elif "Mentoring Tips" in last_response:
                selected = "Mentor"
            elif "Recommended Resources" in last_response:
                selected = "Resource"
    if selected is None:
        yield "Sorry, I couldn't determine which type of support you need. Could you clarify if you want emotional support, planning help, mentoring advice, or resources? "
        return
    # Only run the relevant agent/task
    agents = [AGENT_TASK_MAP[selected][0]]
    tasks = [AGENT_TASK_MAP[selected][1]]
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=False,
        concurrent=True
    )
    output = crew.kickoff(inputs={"topic": user_input, "history": context_str})
    for word in output.raw.split():
        yield word + " "
