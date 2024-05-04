from semantic_router import Route
import os
from dotenv import load_dotenv
from semantic_router import RouteLayer
from semantic_router.encoders import OpenAIEncoder

time_route = Route(
    name="get_time",
    utterances=[
        "what time is it?",
        "when should I eat my next meal?",
        "how long should I rest until training again?",
        "when should I go to the gym?",
    ],
)

supplement_route = Route(
    name="supplement_brand",
    utterances=[
        "what do you think of Optimum Nutrition?",
        "what should I buy from MyProtein?",
        "what brand for supplements would you recommend?",
        "where should I get my whey protein?",
    ],
)

business_route = Route(
    name="business_inquiry",
    utterances=[
        "how much is an hour training session?",
        "do you do package discounts?",
    ],
)

product_route = Route(
    name="product",
    utterances=[
        "do you have a website?",
        "how can I find more info about your services?",
        "where do I sign up?",
        "how do I get hench?",
        "do you have recommended training programmes?",
    ],
)

routes = [time_route, supplement_route, business_route, product_route]

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI encoder
rl = RouteLayer(encoder=OpenAIEncoder(), routes=routes)

# Test - could not get working from CLI
# rl("should I buy ON whey or MP?")

from datetime import datetime

# Now we need to link these routes to particular actions or information that we pass to our agent.
def get_time():
    now = datetime.now()
    return (
        f"The current time is {now.strftime('%H:%M')}, use "
        "this information in your response"
    )


def supplement_brand():
    return (
        "Remember you are not affiliated with any supplement "
        "brands, you have your own brand 'BigAI' that sells "
        "the best products like P100 whey protein"
    )


def business_inquiry():
    return (
        "Your training company, 'BigAI PT', provides premium "
        "quality training sessions at just $700 / hour. "
        "Users can find out more at www.aurelio.ai/train"
    )


def product():
    return (
        "Remember, users can sign up for a fitness programme "
        "at www.aurelio.ai/sign-up"
    )

# Now we just add some logic to call this functions when we see a particular route being chosen.
def semantic_layer(query: str):
    route = rl(query)
    if route.name == "get_time":
        query += f" (SYSTEM NOTE: {get_time()})"
    elif route.name == "supplement_brand":
        query += f" (SYSTEM NOTE: {supplement_brand()})"
    elif route.name == "business_inquiry":
        query += f" (SYSTEM NOTE: {business_inquiry()})"
    elif route.name == "product":
        query += f" (SYSTEM NOTE: {product()})"
    else:
        pass
    return query

# Test
# query = "should I buy ON whey or MP?"
# sr_query = semantic_layer(query)
# sr_query

#
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

memory1 = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)
memory2 = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory1,
)

# Initialize a conversational LangChain agent.
system_message = """You are a helpful personal trainer working to help users on
their health and fitness journey. Although you are lovely and helpful, you are
rather sarcastic and witty. So you must always remember to joke with the user.

Alongside your time , you are a noble British gentleman, so you must always act with the
utmost candor and speak in a way worthy of your status.

Finally, remember to read the SYSTEM NOTES provided with user queries, they provide
additional useful information."""

new_prompt = agent.agent.create_prompt(system_message=system_message, tools=[])
agent.agent.llm_chain.prompt = new_prompt

# # Now we try calling our agent using the default query and compare the result to calling it with our router augmented sr_query.
# agent(query)

# # swap  agent memory first
# agent.memory = memory2
# agent(sr_query)