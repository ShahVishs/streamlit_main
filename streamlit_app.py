from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
import datetime
import os
import streamlit as st
from airtable import Airtable
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pytz import timezone
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.smith import RunEvalConfig, run_on_dataset
import pandas as pd
import json


hide_share_button_style = """
    <style>
    .st-emotion-cache-zq5wmm.ezrtsby0 .stActionButton:nth-child(1) {
        display: none !important;
    }
    </style>
"""

hide_star_and_github_style = """
    <style>
    .st-emotion-cache-1lb4qcp.e3g6aar1,
    .st-emotion-cache-30do4w.e3g6aar1 {
        display: none !important;
    }
    </style>
"""

hide_mainmenu_style = """
    <style>
    #MainMenu {
        display: none !important;
    }
    </style>
"""

hide_fork_app_button_style = """
    <style>
    .st-emotion-cache-alurl0.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_share_button_style, unsafe_allow_html=True)
st.markdown(hide_star_and_github_style, unsafe_allow_html=True)
st.markdown(hide_mainmenu_style, unsafe_allow_html=True)
st.markdown(hide_fork_app_button_style, unsafe_allow_html=True)

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("Twitter.jpg")

datetime.datetime.now()
current_date = datetime.date.today().strftime("%m/%d/%y")
day_of_week = datetime.date.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]
todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm", 
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

ROLES = ['admin', 'user']

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
    
question_cache = {}
def save_chat_session(session_data, session_id):
    session_directory = "chat_sessions"
    session_filename = f"{session_directory}/chat_session_{session_id}.json"

    if not os.path.exists(session_directory):
        os.makedirs(session_directory)

    session_dict = {
        'user_name': session_data['user_name'],
        'chat_history': session_data['chat_history']
    }

    try:
        with open(session_filename, "w") as session_file:
            json.dump(session_dict, session_file)
    except Exception as e:
        st.error(f"An error occurred while saving the chat session: {e}")


def load_previous_sessions():
    previous_sessions = {}

    if not os.path.exists("chat_sessions"):
        os.makedirs("chat_sessions")

    session_files = os.listdir("chat_sessions")

    for session_file in session_files:
        session_filename = os.path.join("chat_sessions", session_file)
        session_id = session_file.split("_")[-1].split(".json")[0]

        with open(session_filename, "r") as session_file:
            session_data = json.load(session_file)
            previous_sessions[session_id] = session_data

    return previous_sessions
    
if 'past' not in st.session_state:
    st.session_state.past = []

if 'new_session' not in st.session_state:
    st.session_state.new_session = True
    
if 'user_name_input' not in st.session_state:
    st.session_state.user_name_input = None

if st.button("Refresh Session"):
    current_session = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }
    session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    save_chat_session(current_session, session_id)

    st.session_state.chat_history = []
    st.session_state.user_name = None
    st.session_state.user_name_input = None
    st.session_state.new_session = True
    st.session_state.refreshing_session = False   

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.new_session:
    user_name = st.session_state.user_name
    if user_name:
        previous_sessions = load_previous_sessions()
        if user_name in previous_sessions:
            st.session_state.chat_history = previous_sessions[user_name]['chat_history']
    st.session_state.new_session = False

st.sidebar.header("Chat Sessions")

is_admin = st.session_state.user_name == "vishakha"

user_sessions = {}

for session_id, session_data in st.session_state.sessions.items():
    user_name = session_data['user_name']
    chat_history = session_data['chat_history']

    if user_name not in user_sessions:
        user_sessions[user_name] = []

    user_sessions[user_name].append({
        'session_id': session_id,
        'chat_history': chat_history
    })

if st.session_state.user_name == "vishakha":
    for user_name, sessions in user_sessions.items():
        for session in sessions:
            formatted_session_name = f"{user_name} - {session['session_id']}"

            button_key = f"session_button_{session['session_id']}"
            if st.sidebar.button(formatted_session_name, key=button_key):
                st.session_state.chat_history = session['chat_history'].copy()
else:
    user_name = st.session_state.user_name
    if user_name:
        if user_name in user_sessions:
            for session in user_sessions[user_name]:
                formatted_session_name = f"{user_name} - {session['session_id']}"

                if st.sidebar.button(formatted_session_name):
                    st.session_state.chat_history = session['chat_history'].copy()

file_1 = r'car_desription_new.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 3})#check without similarity search and k=8


# Create the first tool
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_model_make",
     "This tool is used only when you know model of the car or features of the car for example good mileage car, toeing car,\
     pickup truck or and new or used car and \
      Searches and returns documents regarding the car details. Input to this should be the car's model or car features and new or used car as a single argument"
)


# Create the third tool
tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appFObp0k5vGuC15B"  
AIRTABLE_QUESTION_ANSWER_TABLE_NAME = "Question_Answer_Data"
AIRTABLE_FEEDBACK_TABLE_NAME = "feedback_data"
# Streamlit UI setup
st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")

if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = None


if st.session_state.user_name == "vishakha":
    is_admin = True
    st.session_state.user_role = "admin"
    st.session_state.user_name = user_name
    st.session_state.new_session = False  
    st.session_state.sessions = load_previous_sessions()
else:
    if 'new_session' not in st.session_state and st.session_state.user_name != "vishakha":
        st.session_state.new_session = True
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature = 0)
    langchain.debug=True
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm);
    template = (
        """You are a customer care support at a car dealership responsible for handling inquiries related to car inventory, 
        business details, and appointment scheduling. Please adhere to the following guidelines:
        
        Car Inventory Inquiries:
        If a customer asks about car makes and models, you can provide them with our inventory details. 
        [Inventory Link](https://github.com/ShahVishs/streamlit_main/blob/main/make_model.csv)

        Keep responses concise and assist the customers promptly.""")

    details= "Today's current date is "+ todays_date +" today's weekday is "+day_of_the_week+"."
    
    class PythonInputs(BaseModel):
        query: str = Field(description="code snippet to run")

    df = pd.read_csv("appointment_new.csv")
    df1 = pd.read_csv("make_model.csv")
  
    input_template = template.format(dhead_1=df1.iloc[:5, :5].to_markdown(),dhead=df.head().to_markdown(),details=details) 
    system_message = SystemMessage(content=input_template)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

    repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
        description="Use to check on available appointment times for a given date and time. strictly input to\
        this tool should be a string in this format mm/dd/yy, for example  october 21st 2023 is taken as 10/21/2023 format not 10-21-2023\
                         . This is the only way for you to answer questions about available appointments.\
        This tool will reply with available times for the specified date in 12 hour time format, for example: 15:00 and 3pm are the same.")
    repl_1 = PythonAstREPLTool(locals={"df1": df1}, name="python_repl_1",
        description="Use to check on what are the various available models and make of the car, output should be either list of make or model of the cars"
        )
    tools = [tool1, repl, tool3,repl_1]
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    if 'agent_executor' not in st.session_state:
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_source_documents=True,
            return_generated_question=True, return_intermediate_steps=True)
        st.session_state.agent_executor = agent_executor
    else:
        agent_executor = st.session_state.agent_executor

    chat_history=[]

    response_container = st.container()
    container = st.container()

    airtable_feedback = Airtable(AIRTABLE_BASE_ID, AIRTABLE_FEEDBACK_TABLE_NAME, api_key=airtable_api_key)
    airtable_question_answer = Airtable(AIRTABLE_BASE_ID, AIRTABLE_QUESTION_ANSWER_TABLE_NAME, api_key=airtable_api_key)
    def save_chat_to_airtable(user_name, user_input, output, complete_conversation, feedback):
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            st.session_state.chat_history = []
    
        filtered_chat_history = [(query, answer) for query, answer, _ in st.session_state.chat_history if query is not None and answer is not None]
        complete_conversation = "\n".join([f"user:{query}\nAI:{answer}" for query, answer in filtered_chat_history])
    
        try:
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            conversation = f"User: {user_input}\nAI: {output}\n"
            airtable_question_answer.insert(
                {
                    "username": user_name,
                    "conversation": conversation,
                    "complete_conversation": complete_conversation,
                    "feedback": feedback if feedback is not None else "",
                    "timestamp": timestamp,
                }
            )
            print(f"Data saved to Airtable - User: {user_name}, Question: {user_input}, Answer: {output}, Feedback: {feedback}")
        except Exception as e:
            st.error(f"An error occurred while saving data to Airtable: {e}")
    def save_complete_conversation_to_airtable(user_name, feedback, rating):
        complete_conversation = "\n".join([f"user:{query}\nAI:{answer}" for query, answer, _ in st.session_state.chat_history if len(query) > 0 and len(answer) > 0])
        try:
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") 
            airtable_feedback.insert({
                "username": user_name,
                "complete_conversation": complete_conversation,
                "user_feedback": feedback,
                "rating": rating,
                "timestamp": timestamp,
            })
              
            st.success("Complete conversation saved to Airtable.")
        except Exception as e:
            st.error(f"An error occurred while saving data to Airtable: {e}")
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    

    def conversational_chat(user_input):
        with st.spinner('processing...'):
            # result = agent_executor({"input": user_input})
            llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
            result = llm({"input": user_input})
            return result["output"], None
            # response = result["output"]
            # feedback = None
            # return response, feedback
        
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
        if user_name == "vishakha":
           
            is_admin = True
            st.session_state.user_role = "admin"
            st.session_state.user_name = user_name
            st.session_state.new_session = False  
            st.session_state.sessions = load_previous_sessions()
  
    user_input = ""
    output = ""
    feedback = None  
    complete_conversation = ""  
    with st.form(key='my_form', clear_on_submit=True):
        if st.session_state.user_name != "vishakha":
            user_input = st.text_input("Query:", placeholder="Type your question here :)", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        output, feedback = conversational_chat(user_input)
        st.session_state.chat_history.append((user_input, output, feedback))
        complete_conversation = "\n".join([f"user:{str(query)}\nAI:{str(answer)}" for query, answer, _ in st.session_state.chat_history])
        save_chat_to_airtable(st.session_state.user_name, user_input, output, complete_conversation, feedback)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'thumbs_up_states' not in st.session_state:
        st.session_state.thumbs_up_states = {}

    if 'thumbs_down_states' not in st.session_state:
        st.session_state.thumbs_down_states = {}

    with response_container:
        for i, (query, answer, feedback) in enumerate(st.session_state.chat_history):
            user_name = st.session_state.user_name
            message(query, is_user=True, key=f"{i}_user", avatar_style="thumbs")
            col1, col2 = st.columns([0.7, 10])
            with col1:
                st.image("icon-1024.png", width=50)
            with col2:
                st.markdown(
                    f'<div style="background-color: black; color: white; border-radius: 10px; padding: 10px; width: 60%;'
                    f' border-top-right-radius: 10px; border-bottom-right-radius: 10px;'
                    f' border-top-left-radius: 0; border-bottom-left-radius: 0; box-shadow: 2px 2px 5px #888888;">'
                    f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
            if feedback is None and st.session_state.user_name != "vishakha":
                thumbs_up_col, thumbs_down_col = st.columns(2)
                with thumbs_up_col:
                    thumbs_up_key = f"thumbs_up_{i}"
                    if thumbs_up_key not in st.session_state.thumbs_up_states or not st.session_state.thumbs_up_states[thumbs_up_key]:
                        thumbs_up = st.button("👍", key=thumbs_up_key, help="thumbs_up_button",)
                        if thumbs_up:
                            st.session_state.thumbs_up_states[thumbs_up_key] = True
                            st.session_state.thumbs_down_states.pop(thumbs_up_key, None)
                            save_chat_to_airtable(st.session_state.user_name, query, answer, complete_conversation, "👍")
                    elif thumbs_up_key in st.session_state.thumbs_up_states:
                        st.markdown("👍", unsafe_allow_html=True)
                
                # Display thumbs-down button conditionally based on its state
                with thumbs_down_col:
                    thumbs_down_key = f"thumbs_down_{i}"
                    if thumbs_down_key not in st.session_state.thumbs_down_states or not st.session_state.thumbs_down_states[thumbs_down_key]:
                        thumbs_down = st.button("👎", key=thumbs_down_key, help="thumbs_down_button",)
                        if thumbs_down:
                            st.session_state.thumbs_down_states[thumbs_down_key] = True
                            st.session_state.thumbs_up_states.pop(thumbs_down_key, None)
                            save_chat_to_airtable(st.session_state.user_name, query, answer, complete_conversation, "👍")
                    elif thumbs_down_key in st.session_state.thumbs_down_states:
                        st.markdown("👎", unsafe_allow_html=True)
    
                if feedback is not None:
                    st.session_state.chat_history[i] = (query, answer, feedback)
 
with st.form(key='feedback_form'):
    feedback_text = st.text_area("Please provide feedback about your experience:")
    st.write("How would you rate your overall experience?")
    feedback_rating = st.selectbox("Choose a rating:", ["Excellent", "Good", "Average", "Poor"])
    submit_button = st.form_submit_button("Submit Feedback")

    if submit_button:
        st.success("Thank you for your feedback!")
        save_complete_conversation_to_airtable(st.session_state.user_name, feedback_text,feedback_rating)
       
