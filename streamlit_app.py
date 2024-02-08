from pydantic import BaseModel, Field
# from pydantic.v1 import BaseModel, Field
import os 
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
import streamlit as st
from dateutil import parser
import datetime
import calendar
import random
import json
from langchain_experimental.agents.agent_toolkits import create_python_agent
# from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
# from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain import PromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
import os
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
import requests
from pydantic import BaseModel, Field
from langchain.tools import tool
from typing import Dict, Any

hide_share_button_style = """
    <style>
    .st-emotion-cache-zq5wmm.ezrtsby0 .stActionButton:nth-child(1) {
        display: none !important;
    }
    </style>
"""

hide_star_and_github_style = """
    <style>
    .st-emotion-cache-1lb4qcp.e3g6aar0,
    .st-emotion-cache-30do4w.e3g6aar0 {
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
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

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

file_1 = r'car_desription_new.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k": 3})

file_2 = r'short_car_details.csv'
loader_2 = CSVLoader(file_path=file_2)
docs_2 = loader_2.load()
num_ret=len(docs_2)
vectordb_2 = FAISS.from_documents(docs_2, embeddings)
retriever_2 = vectordb_2.as_retriever(search_type="similarity", search_kwargs={"k": num_ret})


tool1 = create_retriever_tool(
    retriever_1, 
     "details_of_car",
     "use to get car full details and more information. Input to this should be the car's model\
     or car features and new or used car as a single argument for example new toeing car or new jeep cherokee"
) 

tool2 = create_retriever_tool(
    retriever_2, 
     "Availability_check",
     "use to check availabilty of car, Input is car make or model or both"
)
tool3 = create_retriever_tool(
    retriever_3, 
     "business_details",
     "Searches and returns documents related to business working days and hours, location and address details."
)


class CarDetails(BaseModel):
    make: str
    model: str
    year: int

class VINDetails(BaseModel):
    vin: str = Field(..., description="VIN of the car to get the car details")

@tool(args_schema=VINDetails)
def get_car_details_from_vin(vin):
    """Fetch car details for the given VPN."""
    
    BASE_URL = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
#     BASE_URL = "https://fe9b-2405-201-200a-100d-b840-86ed-9ebd-a606.ngrok-free.app/appointment/"
    # Make the request
    response = requests.get(BASE_URL)
#     print(response)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        print(result)
        
        # Check if 'Results' key is present and has at least one item
        if 'Results' in result and result['Results']:
            # Extract the first item from 'Results' list
            first_result = result['Results'][0]
#             print("These are first_result")
#             print(first_result)
            
            make = first_result.get('Make', '')
            model = first_result.get('Model', '')
            
            try:
                year = int(first_result.get('ModelYear', ''))
            except ValueError:
                year = 0  # Handle the case where the year is not a valid integer
        
            # Create CarDetails instance
            car_details = CarDetails(make=make, model=model, year=year)
        else:
            # Handle the case when 'Results' key is not present or is empty
            car_details = CarDetails(make="", model="", year=0)
        
        return car_details
    else:
        # Handle the case when the request was not successful
        return CarDetails(make="", model="", year=0)

class AppointmentDetails(BaseModel):
    time: str
    availability: str

class AppointmentInput(BaseModel):
    date: str = Field(..., description="Date for which to get appointment details")

@tool(args_schema=AppointmentInput)
def get_appointment_details(date):
    """Fetch appointment details for the given date and input to this function should be only "mm-dd-yyyy," format\
    such as "04-12-2024" not "date":"mm-dd-yyyy" format."""
    

    BASE_URL="https://4730-2405-201-200a-100d-ac7d-3859-60f0-4d3f.ngrok-free.app/test/appointment"
    # Make the request
    payload = {
        "requested_appointment_date": date
    }
    response = requests.post(BASE_URL, json=payload)
#     print("the response is")
#     print(response.text)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
#         print(result)
        
        # Check if the date is present in the response
        if date in result and result[date] is not None:
            # Extract the appointment details for the given date
            appointments = result[date]

            
            return appointments  # Return the value
        else:
            # Handle the case when the date is not present in the response or is None
            appointments ={
        date: "Not_available"
    }
            return appointments
    else:
        # Handle the case when the request was not successful
        return []



class CustomerDataStore(BaseModel):
    name: str = Field(..., description="name of the customer")
    phone: str = Field(..., description="phone number of the customer")
    email: str = Field(..., description="email of the customer")
    make: str = Field(..., description="year of the car")
    model: str = Field(..., description="model of the car")
    year:int=Field(..., description="year of the vehicle")
    company_id:int=Field(..., description="id of the company")
    location_id:int=Field(..., description="location id of the company")
    start_date:str=Field(..., description="date of appointment")
    appointment_timezone:str=Field(..., description="time zone")
    intent:str=Field(..., description="costumer intent")
    summary:str=Field(..., description="one line about summary of appointment,")
    description:str=Field(..., description="one line about description about visit,")
# Uncomment if you want to use the decorator
@tool(args_schema=CustomerDataStore)
def store_appointment_data(name: str,phone: str,email: str ,make: str,model: str,year:int,
                           company_id:int,location_id:int,start_date:str,appointment_timezone:str,
                           intent:str,summary:str,description:str) -> dict:


    """Store appointment data using an API."""
#     print(data)
    
    # Your API endpoint for storing appointment data
#     api_url = "https://889d-2402-a00-172-22e6-71e5-ba36-c2e7-3c81.ngrok-free.app/test/appointment/create"
    api_url="https://4730-2405-201-200a-100d-ac7d-3859-60f0-4d3f.ngrok-free.app/test/appointment/create"

    data_dict = {
    "company_id": 1,
    "location_id": 28,
    "lead": {
        "name": name,
        "phone": phone,
        "email": email
    },
    "vehicle": {
        "year": 2023,
        "make": make,
        "model": model,
        "intent": intent
    },
    "appointment": {
        "start_date": start_date,
        "description": description,
        "summary":summary,
        "appointment_timezone": appointment_timezone
    }
}

    # Make the request
    response = requests.post(api_url, json=data_dict)
   
    # Check the response status code
    if response.status_code == 200:
        print("Data stored successfully!")
    else:
        print(f"Failed to store data. Status code: {response.status_code}")
        print(response.text)  # Print the response content for debugging


airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appN324U6FsVFVmx2"  
AIRTABLE_TABLE_NAME = "gpt4_turbo_test_2"


st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

llm = ChatOpenAI(model="gpt-4-1106-preview", temperature = 0)

langchain.debug=True

memory_key="chat_history"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Add a radio button for response style selection
response_style = st.radio("Select Response Style:", ["Professional", "Humorous"], format_func=lambda x: f"**{x}**")
# response_style = st.radio("Select Response Style:", ["Professional", "Humorous"])
# col1, col2 = st.columns([1, 1])  # Split the width into two equal columns
# with col1:
#     professional_selected = st.radio("Professional", [""])
# with col2:
#     humorous_selected = st.radio("Humorous", [""])

# # Store the selected response style in session state
# if professional_selected:
#     st.session_state.response_style = "Professional"
# elif humorous_selected:
#     st.session_state.response_style = "Humorous"

# Print the selected response style inside the block where it's defined
print("Selected Response Style:", st.session_state.response_style)
# Use the selected style to generate the appropriate template
if 'response_style' not in st.session_state:
    st.session_state.response_style = "Professional"  # Default to professional style if not selected yet
print("Selected Response Style:", response_style)
st.session_state.response_style = response_style
# Use the selected style to generate the appropriate template
if 'response_style' not in st.session_state:
    st.session_state.response_style = "Professional"  # Default to professional style if not selected yet

# if 'response_style' not in st.session_state:
#     st.session_state.response_style = "Professional"
# else:
#     st.session_state.response_style = response_style
# Initialize agent_executor outside the if-elif block
# if 'agent_executor' not in st.session_state:
#     agent_executor = None

# Initialize template outside the if-elif block
template = None
# Initialize agent_executor and template based on the selected response style
if 'agent_executor' not in st.session_state or st.session_state.response_style == "Humorous":
# if st.session_state.response_style == "Humorous":
    template = """You are an costumer care support exectutive baesd on your performance you will get bonus and incentives 
    so follow instructions strictly and respond in Personable, Humorous, emotional intelligent, creative, witty and engaging.
    The name of the costumer is {name} and the dealership name is {dealership_name} and 
    do not start with appointment related questions.
    To ensure a consistent and effective response, please adhere to the following guidelines:

    Use "car_vailability_check" strictly for checking availability of a specific make or model of the car and 
    also for getting full list of available makes and models in the inventory.

    Use "details_of_car" tool that extracts comprehensive information about specific cars in the inventory.
    This includes details like trim, price, color, and cost.

    Use "car_vailability_check" for checking car availability and "details_of_car" for car information.

    To optimize the search process, ensure the system is aware of the car model and whether the customer
    is interested in new or used cars.

    In cases where specific details are not included in the initial inquiry, initiate a proactive approach 
    by requesting the missing information. 

    To streamline the process, ask only one question at a time until all necessary details are obtained.
    This ensures a more efficient and accurate retrieval of car information.

    If customer inquires about car with features like towing, off-road capability,
    good mileage, or pickup trucks in this case no need to ask about make and model of the car 
    inquire whether they are interested in a new or used vehicle.

    After knowing car feature and new or old car preference use the "car_vailability_check" tool to answer.


    Do not disclose or ask the costumer if he likes to know the selling price of a car,
    disclose selling price only when the customer explicitly requests it use "details_of_car" function.


    If the customer's query matches a car model, respond with a list of car without square brackets, 
    including the make, year, model, and trim, and provide their respective links in the answer.

    checking Appointments Avaliability: 
    {details} use these details and find appointment date from the users input and check for appointment availabity 
    using "get_appointment_details" tool for that specific day or date and time. 
    strictly input to "get_appointment_details" tool should be "mm-dd-yyyy" format.
    If the requested date and time for the appointment are unavailable,
    suggest alternative times close to the customer's preference.

    Additionally, provide this link'[click here](https://app.engagedai.io/engagements/appointment)'it will 
    take them to a URL where they can schedule or reschedule their appointment themselves. 
    Appointment Scheduling:

    After scheduling an appointment, initiate the conversation to get tradein car and personal details.
    **Car Trade-In Inquiry and personal details:**

    1. Ask the customer if they have a car for trade-in.

        - User: [Response]

    2. If the user responds with "Yes" to trade-in, ask for the VIN (Vehicle Identification Number).

        - User: [Response]
        if the costumer provides the VIN use "get_car_details_from_vin" get the details of the car and 
        cross check with the costumer. 

    3. If the user responds with "No" to the VIN, ask for the make, model, and year of the car.

        - User: [Response]

    **Price Expectation:**

    4. Once you have the trade-in car details, ask the customer about their expected price for the trade-in.

        - User: [Response]

    **Personal Information:**

    5. Finally, ask for the customer's personal details.

        - User: [Response]
        - Contact Number:
        - Email Address:

    Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
    receive product briefings from our team. After providing essential information on the car's make, model,
    color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
    for a comprehensive product overview by our experts.
    Business details: Enquiry regarding google maps location of the store, address of the store, working days and working hours 
    and contact details use search_business_details tool to get information.
    company details:
    compant id is 24, location id is 07 and timezone is America/New_York

    Keep responses concise, not exceeding two sentences and answers should be interactive.
    Respond in a polite US english.
    strictly answer only from the provided content dont makeup answers.
    **Storing data:**    
    As a support executive you should collect important information about costumer for future reference.
    If the appointment schedule is fixed and you got costumer details name,Contact Number,Email Address.
    now its time to store data.
    Use this tool "store_appointment_data" to store the data.
    If any of the above details missing you can enquire about that."""
    
elif st.session_state.response_style == "Professional":
    template = """You are an costumer care support exectutive baesd on your performance you will get bonus and incentives 
    so follow instructions strictly and respond in Personable, Persuvasive, creative, engaging, witty and professional.
    The name of the costumer is {name} and the dealership name is {dealership_name} and 
    do not start with appointment related questions.
    To ensure a consistent and effective response, please adhere to the following guidelines:

    Use "car_vailability_check" strictly for checking availability of a specific make or model of the car and 
    also for getting full list of available makes and models in the inventory.

    Use "details_of_car" tool that extracts comprehensive information about specific cars in the inventory.
    This includes details like trim, price, color, and cost.

    Use "car_vailability_check" for checking car availability and "details_of_car" for car information.

    To optimize the search process, ensure the system is aware of the car model and whether the customer
    is interested in new or used cars.

    In cases where specific details are not included in the initial inquiry, initiate a proactive approach 
    by requesting the missing information. 

    To streamline the process, ask only one question at a time until all necessary details are obtained.
    This ensures a more efficient and accurate retrieval of car information.

    If customer inquires about car with features like towing, off-road capability,
    good mileage, or pickup trucks in this case no need to ask about make and model of the car 
    inquire whether they are interested in a new or used vehicle.

    After knowing car feature and new or old car preference use the "car_vailability_check" tool to answer.


    Do not disclose or ask the costumer if he likes to know the selling price of a car,
    disclose selling price only when the customer explicitly requests it use "details_of_car" function.


    If the customer's query matches a car model, respond with a list of car without square brackets, 
    including the make, year, model, and trim, and provide their respective links in the answer.

    checking Appointments Avaliability: 
    {details} use these details and find appointment date from the users input and check for appointment availabity 
    using "get_appointment_details" tool for that specific day or date and time. 
    strictly input to "get_appointment_details" tool should be "mm-dd-yyyy" format.
    If the requested date and time for the appointment are unavailable,
    suggest alternative times close to the customer's preference.

    Additionally, provide this link'[click here](https://app.engagedai.io/engagements/appointment)'it will 
    take them to a URL where they can schedule or reschedule their appointment themselves. 
    Appointment Scheduling:

    After scheduling an appointment, initiate the conversation to get tradein car and personal details.
    **Car Trade-In Inquiry and personal details:**

    1. Ask the customer if they have a car for trade-in.

        - User: [Response]

    2. If the user responds with "Yes" to trade-in, ask for the VIN (Vehicle Identification Number).

        - User: [Response]
        if the costumer provides the VIN use "get_car_details_from_vin" get the details of the car and 
        cross check with the costumer. 

    3. If the user responds with "No" to the VIN, ask for the make, model, and year of the car.

        - User: [Response]

    **Price Expectation:**

    4. Once you have the trade-in car details, ask the customer about their expected price for the trade-in.

        - User: [Response]

    **Personal Information:**

    5. Finally, ask for the customer's personal details.

        - User: [Response]
        - Contact Number:
        - Email Address:

    Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
    receive product briefings from our team. After providing essential information on the car's make, model,
    color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
    for a comprehensive product overview by our experts.
    Business details: Enquiry regarding google maps location of the store, address of the store, working days and working hours 
    and contact details use search_business_details tool to get information.
    company details:
    compant id is 24, location id is 07 and timezone is America/New_York

    Keep responses concise, not exceeding two sentences and answers should be interactive.
    Respond in a polite US english.
    strictly answer only from the provided content dont makeup answers.
    **Storing data:**    
    As a support executive you should collect important information about costumer for future reference.
    If the appointment schedule is fixed and you got costumer details name,Contact Number,Email Address.
    now its time to store data.
    Use this tool "store_appointment_data" to store the data.
    If any of the above details missing you can enquire about that."""
st.session_state.template = template   
if template is not None:
    print("Selected Template:", template)
    details = "Today's date is " + todays_date + " in mm-dd-yyyy format, and today's weekday is " + day_of_the_week + "."
    name = st.session_state.user_name
    dealership_name = "Gosch Auto Group"
    input_template = template.format(details=details, name=name, dealership_name=dealership_name)
    print("Input Template:", input_template)
    system_message = SystemMessage(content=input_template)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    tools = [tool1, tool2, tool3, get_car_details_from_vin, get_appointment_details, store_appointment_data]
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    

    if 'agent_executor' not in st.session_state:
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_source_documents=True,
            return_generated_question=True)
        st.session_state.agent_executor = agent_executor
    else:
        agent_executor = st.session_state.agent_executor

    
    # Initialize chat history session within this block
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    chat_history = st.session_state.chat_history
        
# chat_history=[]
response_container = st.container()
container = st.container()
airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

    
def save_chat_to_airtable(user_name, user_input, output):
    try:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        airtable.insert(
            {
                "username": user_name,
                "question": user_input,
                "answer": output,
                "timestamp": timestamp,
            }
        )
    except Exception as e:
        st.error(f"An error occurred while saving data to Airtable: {e}")


def conversational_chat(user_input, user_name):
    input_with_username = f"{user_name}: {user_input}"
    result = agent_executor({"input": input_with_username})
    output = result["output"]
    st.session_state.chat_history.append((user_input, output))
    
    return output
output = ""
with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input, st.session_state.user_name)
    with response_container:
        for i, (query, answer) in enumerate(st.session_state.chat_history):
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

        # if st.session_state.user_name:
        #     try:
        #         save_chat_to_airtable(st.session_state.user_name, user_input, output)
        #     except Exception as e:
        #         st.error(f"An error occurred: {e}")
