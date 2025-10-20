from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool, LiteLLMModel, TransformersModel

def simple_agent_example():
    # Initialize a model (using Hugging Face Inference API)
    model = InferenceClientModel()  # Uses a default model

    # Create an agent with no tools
    agent = CodeAgent(tools=[], model=model)

    # Run the agent with a task
    result = agent.run("Calculate the sum of numbers from 1 to 10")
    print(result)


def agent_with_search_example():
    # Initialize a model (using Hugging Face Inference API)
    model = InferenceClientModel()  # Uses a default model

    # Create a search tool
    search_tool = DuckDuckGoSearchTool()

    # Create an agent with the search tool
    agent = CodeAgent(tools=[search_tool], model=model)

    # Run the agent with a task that requires searching
    result = agent.run("What is the capital of France?")
    print(result)


def agent_with_LiteLLMModel():
    # Using OpenAI/Anthropic (requires 'smolagents[litellm]')
    model = LiteLLMModel(model_id="gpt-4")


    # Create a search tool
    search_tool = DuckDuckGoSearchTool()

    # Create an agent with the search tool
    agent = CodeAgent(tools=[search_tool], model=model)

    # Run the agent with a task that requires searching
    result = agent.run("What is the capital of France?")
    print(result)    


def agent_with_transformers_model():
    # Using local models (requires 'smolagents[transformers]')
    model = TransformersModel(model_id="meta-llama/Llama-2-7b-chat-hf")

    # Create a search tool
    search_tool = DuckDuckGoSearchTool()

    # Create an agent with the search tool
    agent = CodeAgent(tools=[search_tool], model=model)

    # Run the agent with a task that requires searching
    result = agent.run("What is the capital of France?")
    print(result)


if __name__ == "__main__":
    # simple_agent_example()
    # agent_with_search_example()
    # agent_with_LiteLLMModel()     # need api key!!!
    agent_with_transformers_model()   # wait for meta's approval!!!