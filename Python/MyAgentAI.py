from dotenv import load_dotenv
import os
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()
search_tool = SerperDevTool()

# Define agents with role and goals
researcher = Agent(
    role = "Senior research assistant",
    goal = "Look up the latest advancements in AI Agents",
    backstory = """You work at a leading tech think tank.
    Your expertise lies in searching Google for AI Agent framework.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose = False,
    allow_delegation = False,
    tools = [search_tool],
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2)
)
writer = Agent(
    role = "Professional Short-Article Writer",
    goal = "Summarize the lastest advancement in AI agents in a concise article",
    backstory = """You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose = True,
    allow_delegation = True,
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7)
)

# Define tasks for crew members
task1 = Task(
    description = """Conduct a comprehensive analysis of the latest advancements in AI Agent in March 2024.
    Indentify key trends, breakthrough, and potential industry impacts""",
    expected_output = "Full analysis report in bullet points",
    agent = researcher
)
task2 = Task(
    description = """Using the insights provided, write a short article
    post that highlights the most significent AI Agent advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make is sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output = "Full blog post of at least 3 paragrphs",
    agent = writer
)
# Instantiate your crew with a sequential process
crew = Crew(
    agents = [researcher, writer],
    tasks = [task1,task2],
    verbose = 1 #1 or 2 different logging levels
)

result = crew.kickoff()
print("########################")
print(result)