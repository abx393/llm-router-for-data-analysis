from typing_extensions import Annotated
import pandas as pd
import time

from autogen import (
    ConversableAgent,
    UserProxyAgent,
)

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.targets.transition_target import AgentTarget, RevertToUserTarget
from autogen.agentchat.group import ReplyResult, ContextVariables, ExpressionContextCondition, ExpressionAvailableCondition, ContextExpression, OnContextCondition

from llm_configs import llama_instruct, llama_instruct_3b
from prompt_templates import intermediate_query_prefix, basic_query_prefix

def analyze_request(
    request: Annotated[str, "The user request text to analyze"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Analyze a user request to determine routing based on content
    Updates context variables with routing information
    """
    context_variables["question_answered"] = False

    print('analyze_request time', time.time())

    # Update request tracking
    context_variables["routing_started"] = True
    context_variables["request_count"] += 1
    context_variables["current_request"] = request
    context_variables["request_times"].append(time.time())

    # Previous domain becomes part of history
    if context_variables["current_complexity"]:
        prev_complexity = context_variables["current_complexity"]
        context_variables["previous_complexities"].append(prev_complexity)
        if prev_complexity in context_variables["complexity_history"]:
            context_variables["complexity_history"][prev_complexity] += 1
        else:
            context_variables["complexity_history"][prev_complexity] = 1

    # Reset current_domain to be determined by the router
    context_variables["current_complexity"] = None

    return ReplyResult(
        message=f"Request analyzed. Will determine the best specialist to handle: '{request}'",
        context_variables=context_variables
    )

def route_to_intermediate_analyst(
    confidence: Annotated[int, "Confidence level for Llama Instruct 3B (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to Llama Instruct 3B"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the intermediate analyst
    """
    context_variables["current_complexity"] = "medium"
    context_variables["complexity_confidence"]["medium"] = confidence
    context_variables["llama_instruct_3b_invocations"] += 1
    context_variables["routing_times"].append(time.time())

    return ReplyResult(
        target=AgentTarget(agent=intermediate_data_analyst),
        message=f"Routing to Llama Instruct 3B with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables
    )

def route_to_basic_analyst(
    confidence: Annotated[int, "Confidence level for Llama Instruct (1-10)"],
    reasoning: Annotated[str, "Reasoning for routing to Llama Instruct"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Route the current request to the basic analyst
    """
    context_variables["current_complexity"] = "low"
    context_variables["complexity_confidence"]["low"] = confidence
    context_variables["llama_instruct_invocations"] += 1
    context_variables["routing_times"].append(time.time())

    return ReplyResult(
        target=AgentTarget(agent=basic_data_analyst),
        message=f"Routing to Llama Instruct with confidence {confidence}/10. Reasoning: {reasoning}",
        context_variables=context_variables
    )

def provide_llama_instruct_3b_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from Llama Instruct 3B
    """
    print('Llama Instruct 3B response time', time.time())
    # Record the question and response
    context_variables["question_responses"].append({
        "complexity": "medium",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True
    context_variables["response_times"].append(time.time())

    return ReplyResult(
        message="Llama Instruct 3B response provided.",
        context_variables=context_variables
    )

def provide_llama_instruct_response(
    response: Annotated[str, "The specialist's response to the request"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Submit a response from Llama Instruct
    """

    print('Llama instruct response time', time.time())

    # Record the question and response
    context_variables["question_responses"].append({
        "complexity": "low",
        "question": context_variables["current_request"],
        "response": response
    })
    context_variables["question_answered"] = True
    context_variables["response_times"].append(time.time())

    return ReplyResult(
        message="Llama instruct response provided.",
        context_variables=context_variables
    )

# Create the agents for the routing system
router_agent = ConversableAgent(
    name="router_agent",
    system_message="""You are the routing agent responsible for analyzing user requests and directing them to the most appropriate specialist.
Do not directly answer the question yourself.

Your task is to carefully analyze the complexity of each user query and determine which agent is most appropriate to answer the query:

1. Basic data analyst: For simple queries requiring basic analysis of text.
2. Intermediate data analyst: For queries requiring synthesis of multiple pieces of the input data.

For each query, you must:
1. Use the analyze_request tool to process the query and update context
2. Determine the correct level of complexity of the query by analyzing the input data and the difficulty of the reasoning required to answer the query.
3. Route to the appropriate analyst using the corresponding routing tool.

When routing:
- Provide a confidence level (1-10) based on how certain you are about the difficulty of the query
- Include detailed reasoning for your routing decision
- If the difficulty of the query is ambiguous, route to the most capable agent.

Always maintain context awareness by considering:
- Current query content and intent
- Previously discussed topics
- User's possible follow-up patterns

After an agent has provided an answer, output the question and answer.
""",
    functions=[
        analyze_request,
        route_to_intermediate_analyst,
        route_to_basic_analyst
    ],
    llm_config=llama_instruct
)

intermediate_data_analyst = ConversableAgent(
    name="intermediate_data_analyst",
    system_message="""You are an intermediate data analyst that can answer moderately complex data analysis queries.

When responding to queries, explain reasoning in clear terms appropriate for the user's apparent level of understanding.

Focus on being informative, precise, and helpful.

Use the provide_llama_instruct_3b_response tool to submit your final response.""",
    functions=[provide_llama_instruct_3b_response],
    llm_config=llama_instruct_3b
)

basic_data_analyst = ConversableAgent(
    name="basic_data_analyst",
    system_message="""You are a basic data analyst that can answer simple data analysis queries as quickly as possible.

When responding to queries, explain reasoning in clear terms appropriate for the user's apparent level of understanding.

Focus on being informative, precise, and helpful.

Use the provide_llama_instruct_response tool to submit your final response.""",
    functions=[provide_llama_instruct_response],
    llm_config=llama_instruct
)

# User agent for interaction
user = UserProxyAgent(
    name="user",
    code_execution_config=False
)

# Register handoffs for the context-aware routing pattern
# Router agent to specialists based on difficulty of query
router_agent.register_handoffs(conditions=[
    # Route to intermediate data analyst when query is moderately complex
    OnContextCondition(
        target=AgentTarget(agent=intermediate_data_analyst),
        condition=ExpressionContextCondition(expression=ContextExpression(expression="${current_complexity} == 'medium'")),
        available=ExpressionAvailableCondition(expression=ContextExpression(expression="!${question_answered}"))
    ),
    # Route to basic data analyst when query is simple
    OnContextCondition(
        target=AgentTarget(agent=basic_data_analyst),
        condition=ExpressionContextCondition(expression=ContextExpression(expression="${current_complexity} == 'low'")),
        available=ExpressionAvailableCondition(expression=ContextExpression(expression="!${question_answered}"))
    ),
])
# router_agent.handoffs.set_after_work(target=RevertToUserTarget())

products_df = pd.read_csv('datasets/products.csv')
print(len(products_df))

# Run the context-aware routing pattern
print("Initiating Context-Aware Routing Pattern...")


for row in range(10):
    print(f'Row {row+1}')
    product_description = products_df.loc[row, 'description']
    review_text = products_df.loc[row, 'text']

    print(product_description)
    print(review_text)

    intermediate_query = f"""
{intermediate_query_prefix}
description: \"{product_description}\"
reviewText: \"{review_text}\"
Answer:
    """

    basic_query = f"""
{basic_query_prefix}
ReviewText: \"{review_text}\"
Answer:
    """

    # Shared context for tracking the conversation and routing decisions
    shared_context = ContextVariables(data={
        # Routing state
        "routing_started": False,
        "current_complexity": None,
        "previous_complexities": [],
        "complexity_confidence": {},
        "routing_times": [],

        # Request tracking
        "request_count": 0,
        "current_request": "",
        "complexity_history": {},
        "request_times": [],

        # Response tracking
        "question_responses": [], # List of question-response pairs
        "question_answered": True, # Indicates if the last question was answered
        "response_times": [],

        # Specialist invocation tracking
        "llama_instruct_3b_invocations": 0,
        "llama_instruct_invocations": 0,

        # Error state (not handled but could be used to route to an error agent)
        "has_error": False,
        "error_message": "",
    })

    agent_pattern = DefaultPattern(
        agents=[
            router_agent,
            intermediate_data_analyst,
            basic_data_analyst
        ],
        initial_agent=router_agent,
        context_variables=shared_context,
        user_agent=user,
    )

    chat_result, final_context, last_agent = initiate_group_chat(
        pattern=agent_pattern,
        messages=f"I have a question: {intermediate_query}",
        max_rounds=10,
    )

    # Display the Questions and Answers
    print("\n===== QUESTION-RESPONSE PAIRS =====\n")
    print('num question responses', len(final_context['question_responses']))
    print('num response times', len(final_context['response_times']))
    for i, qr_pair in enumerate(final_context["question_responses"]):
        if i > 0:
            break
        print(f"{i+1}. Complexity: {qr_pair['complexity'].capitalize()}")
        print(f"Question: {qr_pair['question']}")
        print(f"Response: {qr_pair['response']}")
        print(f"Routing time {final_context['routing_times'][i]}")
        print(f"Request time {final_context['request_times'][i]}")
        print(f"Time to Routing Decision (s): {final_context['routing_times'][i] - final_context['request_times'][i]}")
        print(f"Total Time Elapsed (s): {final_context['response_times'][i] - final_context['request_times'][i]}\n\n")

    # Display the results
    print("\n===== REQUEST ROUTING SUMMARY =====\n")
    print(f"Total Requests: {final_context['request_count']}")
    print(f"Predicted Complexity: {final_context['current_complexity']}")

    # Display the routing history
    print("\n===== ROUTING HISTORY =====\n")
    for complexity, count in final_context["complexity_history"].items():
        print(f"{complexity.capitalize()}: {count} time(s)")

    # Show specialist invocation counts
    print("\n===== SPECIALIST INVOCATIONS =====\n")
    print(f"Intermediate Data Analyst: {final_context['llama_instruct_3b_invocations']}")
    print(f"Basic Data Analyst: {final_context['llama_instruct_invocations']}")

    # Display the conversation flow
    print("\n===== SPEAKER ORDER =====\n")
    for message in chat_result.chat_history:
        if "name" in message and message["name"] != "_Group_Tool_Executor":
            print(f"{message['name']}")
