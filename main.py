import streamlit as st
from langchain.llms import LlamaCpp
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
import json


# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def create_chain(system_prompt):
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear as if the LLM is typing the
    # responses in real time.
    # callback_manager = CallbackManager([stream_handler])

    llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q4_0.gguf",
            temperature=0,
            max_tokens=512,
            top_p=1,
            # callback_manager=callback_manager,
            verbose=False,
            streaming=True,
            )

    # Template you will use to structure your user input into before converting
    # into a prompt. Here, my template first injects the personality I wish to give
    # to the LLM before in the form of system_prompt pushing the actual prompt from the user.
    # Then we'll inject the chat history followed by the user prompt and a placeholder token
    # for the LLM to complete.
    template = """
    {}

    {}

    Human: {}
    AI: 
    """.format(system_prompt, "{chat_history}","{human_input}")

    # We create a prompt from the template so we can use it with langchain
    # prompt = ChatPromptTemplate.from_messages([
    #     SystemMessage(content=system_prompt),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     HumanMessagePromptTemplate.from_template("{human_input}")
    #     ])
    prompt = PromptTemplate(input_variables=["chat_history","human_input"], template=template)

    # Conversation buffer memory will keep track of the conversation in the memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # We create an llm chain with our llm with prompt and memory
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=True)

    return llm_chain


# Set the webpage title
st.set_page_config(
    page_title="Your own Chat!"
)

# Create a header element
st.header("Your own Chat!")

# This sets the LLM's personality for each prompt.
# The initial personality privided is basic.
# Try something interesting and notice how the LLM responses are affected.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")

# Create llm chain to use for our chat bot.
llm_chain = create_chain(system_prompt)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Pass our input to the llm chain and capture the final responses.
    # It is worth noting that the Stream Handler is already receiving the
    # streaming response as the llm is generating. We get our response
    # here once the llm has finished generating the complete response.
    response = llm_chain.predict(human_input=user_prompt)

    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)
