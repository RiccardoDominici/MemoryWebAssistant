# =====================
# Web Search Agent Module
# =====================

def run_agent(custom_prompt=None, step=1):
    """
    Runs a web search agent using smolagents tools and a LiteLLMModel.
    Allows for custom prompts and step control.
    Returns the agent's response to the prompt.
    """
    # Import required agent classes and tools from smolagents
    from smolagents import (  # type: ignore
        CodeAgent,
        ToolCallingAgent,
        WebSearchTool,
        VisitWebpageTool,
        LiteLLMModel,
    )

    # Initialize the language model for the agent
    model = LiteLLMModel(model_id="ollama_chat/qwen3:4b")

    # Create a ToolCallingAgent with web search and webpage visit tools
    search_agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        description="This is an agent that can do web search. Answer questions using web search, visit webpages.",
        max_steps=step,  # Limit the number of steps for the agent
        verbosity_level=0,  # Set verbosity to minimal
    )

    # Create a manager agent to coordinate the search agent
    manager_agent = CodeAgent(
        tools=[],  # No direct tools for the manager
        model=model,
        managed_agents=[search_agent],  # The search agent is managed
        provide_run_summary=False,
        max_steps=step,
        verbosity_level=0
    )

    # Use a default prompt if none is provided
    if custom_prompt is None:
        custom_prompt = "Se gli Stati Uniti mantengono il tasso di crescita del 2024, quanti anni ci vorranno perché il PIL raddoppi?"

    # Run the manager agent with the provided prompt
    response = manager_agent.run(custom_prompt)
    return response

# =====================
# Script Entry Point
# =====================

if __name__ == "__main__":
    # Run the agent with the default prompt and print the response
    response = run_agent()
    print(response)

