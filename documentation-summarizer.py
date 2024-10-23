import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Updated import
from langchain.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
import requests
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union
import re
import gradio as gr
from langchain_community.tools import TavilySearchResults

load_dotenv()

# Define a custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if the output indicates a final answer
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Update the template to include instructions for providing a final answer
template = """
You are an AI assistant tasked with analyzing technical documentation and providing a detailed summary.
You have access to the following tools:

{tool_names}

{tools}

When you need to use a tool, output in the following format:

Action: {{tool_name}}
Action Input: {{tool_input}}

For the Tavily Search tool, provide the search query as the Action Input.

Follow these steps to analyze the documentation:

1. Use the Tavily Search tool to find and gather information about the API or service.
2. Use the Content Understanding tool to analyze the gathered information.
3. If you need more details on specific aspects, use the Tavily Search tool again with more focused queries.
4. Repeat steps 2-3 as necessary until you have a comprehensive understanding of the documentation.

Your final summary should include:
1. An overview of the API or service
2. Key features and functionalities
3. Authentication methods
4. Basic steps for integration or usage
5. Important considerations for developers (e.g., rate limits, error handling)
6. Any unique or standout aspects of the API or service

Only provide your final answer when you are confident you have gathered and analyzed all necessary information.
When you're ready to give your final summary, output in the following format:

Final Answer: {{your detailed summary}}

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "agent_scratchpad"],
)

# Content Understanding Tool
class ContentUnderstandingTool(BaseTool):
    name: str = "Content Understanding"
    description: str = "Analyzes and summarizes the content of technical documentation"

    def _run(self, content: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found.")
        llm = ChatOpenAI(temperature=0, api_key=api_key)
        prompt_text = f"""
Analyze the following technical documentation content and provide a summary:

{content}

Your summary should include:
1. A brief overview of the service or API described in the documentation.
2. Key features or functionalities.
3. Basic steps for integration or usage.
4. Any important notes or considerations for developers.

Provide your summary in a clear, concise format.
"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        response = llm(messages)
        return response.content.strip()

    def _arun(self, content: str) -> str:
        raise NotImplementedError("ContentUnderstandingTool does not support async")

def analyze_documentation(url: str):
    # Initialize tools
    tavily_search = TavilySearchResults()
    content_understanding_tool = ContentUnderstandingTool()
    tools = [tavily_search, content_understanding_tool]

    # Initialize LLM with error handling for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    llm = ChatOpenAI(temperature=0, api_key=api_key)

    # Create the agent with the custom prompt
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=CustomOutputParser(),
    )

    # Create the executor to handle the agent's tasks
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate"
    )

    # Run the agent
    result = agent_executor.invoke({"input": f"Analyze the documentation at {url}"})
    return result["output"]

def gradio_interface(url: str):
    try:
        summary = analyze_documentation(url)
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter Documentation URL"),
    outputs=gr.Textbox(label="Analysis Result"),
    title="Documentation Analysis Tool",
    description="Enter a URL for technical documentation, and the AI will analyze and summarize it.",
)

if __name__ == "__main__":
    iface.launch()

# if __name__ == "__main__":
#     documentation_url = "https://developers.miro.com/docs/miro-web-sdk-introduction"
#     summary = analyze_documentation(documentation_url)
#     print("Documentation Summary:")
#     print(summary)
