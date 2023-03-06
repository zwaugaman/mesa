from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
import streamlit as st
from streamlit_chat import message
from styles import inject_ga, contact_form, feedback_form

def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

st.header("CatGPT 🐈")

template = open_file("prompts/persona/catgpt_persona.txt")

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

# Enter your API key
OpenAI_api_key = st.text_input("**Step #1**: Enter in Your OpenAI API Key", help="Instructions on how to get an API key at https://elephas.app/blog/how-to-create-openai-api-keys-cl5c4f21d281431po7k8fgyol0")

if OpenAI_api_key != "":
    # Set up LLM model with API key
    llm = OpenAI(
        temperature=0, 
        model_name='text-davinci-003', 
        openai_api_key=OpenAI_api_key
    )

    # Set up LLM model with API key
    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt, 
        verbose=True, 
        memory=ConversationalBufferWindowMemory(k=2),
    )

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        input_text = st.text_input("**Step #2:** Chat","Why don’t cats play poker in the jungle?", key="input")
        return input_text 

    user_input = get_text()

    if user_input:
        output = chatgpt_chain.predict(human_input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="initials",seed="CG")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
else:
    st.info("Please enter in your OpenAI API Key. \
            You can create an OpenAI API Key with your \
            OpenAI account at https://openai.com/. \
            Please see https://elephas.app/blog/how-to-create-openai-api-keys-cl5c4f21d281431po7k8fgyol0 \
            for more information. ")

st.sidebar.header("WroteScan :necktie:")

st.sidebar.write("\n Feedback? Please feel free to send us an anonymous message")
st.sidebar.markdown(feedback_form, unsafe_allow_html=True)
st.sidebar.write("")

st.sidebar.write("\n\nWant to learn more? Contact us")
st.sidebar.markdown(contact_form, unsafe_allow_html=True)

st.sidebar.write("")
st.sidebar.write("Try AI party games at [www.aicharades.com](https://www.aicharades.com/)")
