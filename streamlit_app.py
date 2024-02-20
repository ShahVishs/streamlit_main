from pydantic import BaseModel, Field
import os 
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
# from langchain.chains import PALChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
# import streamlit as st
from dateutil import parser
# from datetime import datetime
from langchain.document_loaders import JSONLoader
import datetime
import calendar
import random
import re
import json
# from faker import Faker
# from datetime import datetime, timedelta
# from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
# from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain import PromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
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
# import datetime
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
# from datetime import datetime
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
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://maps.app.goo.gl/ecHtb6y5f8q5PUxb9"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()

file_1 = r'inventory.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k": 8})

# file_2 = r'short_car_details.csv'
# loader_2 = CSVLoader(file_path=file_2)
# docs_2 = loader_2.load()
# num_ret=len(docs_2)
# vectordb_2 = FAISS.from_documents(docs_2, embeddings)
# retriever_2 = vectordb_2.as_retriever(search_type="similarity", search_kwargs={"k": num_ret})

# file_3 = r'csvjson.json'
# loader_3 = JSONLoader(file_path=file_3, jq_schema='.', text_content=False)
# data_3 = loader_3.load()
# vectordb_3 = FAISS.from_documents(data_3, embeddings)
# retriever_4 = vectordb_3.as_retriever(search_type="similarity", search_kwargs={"k": 8})


tool1 = create_retriever_tool(
    retriever_1, 
     "details_of_car",
     "use to check availabilty of car and  to get car full details with images. Input to this should be the car's model\
     or car features and new or used car as a single argument for example new toeing car or new jeep cherokee\
     and also use for getting images based on make and model "
) 

# tool2 = create_retriever_tool(
#     retriever_2, 
#      "Availability_check",
#      "use to check availabilty of car, Input is car make or model or both"
# )
tool3 = create_retriever_tool(
    retriever_3, 
     "business_details",
     "Searches and returns documents related to business working days and hours, location and address details."
)

# tool4 = create_retriever_tool(
#     retriever_4, 
#      "image_details",
#      "Use to search for vehicle information and images based on make and model."
# )



########VIN_FUNCTION########
class CarDetails(BaseModel):
    make: str
    model: str
    year: int

class VINDetails(BaseModel):
    vin: str = Field(..., description="VIN of the car to get the car details")

@tool
def get_car_details_from_vin(vin):
    """Fetch car details for the given VIN."""
    
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


#######APPOINTMENT DATE TO CHECK AVAILABILITY##############

class AppointmentDetails(BaseModel):
    time: str
    availability: str

class AppointmentInput(BaseModel):
    requested_appointment_date: str = Field(..., description="Date for which to get appointment details")
    company_id: int = Field(..., description="company ID")
    location_id: int = Field(..., description="location of dealership")
        
@tool
def check_appointment_availability(requested_appointment_date: str, company_id: int, location_id: int) -> dict:
# def get_appointment_details(requested_appointment_date: str):
    """This tool checks appointment availability for the date that costumer has prefered. Input to this is costumers prefered date."""
    
    BASE_URL = "https://webapp-api-green.prod.funnelai.com/test/appointment"
    
    # Make the request
    payload = {
        "requested_appointment_date": requested_appointment_date,
        "company_id": company_id,
        "location_id": location_id
    }

    response = requests.get(BASE_URL, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        
        # Check if the date is present in the response
        if requested_appointment_date in result and result[requested_appointment_date] is not None:
            # Extract the appointment details for the given date
            appointments = result[requested_appointment_date]
            return appointments
        else:
            # Handle the case when the date is not present in the response or is None
            return {requested_appointment_date: "Not available"}
    else:
        # Handle the case when the request was not successful
        return {"error": "Failed to retrieve appointment details"}




##########DATE_NOT_KNOWN#####CREATING_APPOINTMENT_LINK######
import requests
from pydantic import BaseModel, Field
from typing import Dict, Any

class appointment_link(BaseModel):
    appointment_url: str
#     link:str
class CustomerDataStore(BaseModel):
    name: str = Field(..., description="name of the customer")
    phone: str = Field(..., description="phone number of the customer")
    email: str = Field(..., description="email of the customer")
    make: str = Field(..., description="make or makes of the car")
    model: str = Field(..., description="a single car or a multiple models of the car with comma separated")
    year:int=Field(..., description="year of the vehicle")
    company_id:int=Field(..., description="id of the company")
    location_id:int=Field(..., description="location id of the company")
    start_date:str=Field(..., description="this should be empty string")
    appointment_timezone:str=Field(..., description="time zone")
    intent:str=Field(..., description="costumer intent")
    summary:str=Field(..., description="one line about summary of appointment,")
    description:str=Field(..., description="one line about description about visit,")

@tool
def create_appointment_link(name: str,phone: str,email: str ,make: str,model: str,year:int,
                           company_id:int,location_id:int,start_date:str,appointment_timezone:str,
                           intent:str,summary:str,description:str) -> dict:



    """This tool is used to create appointment link when costumer is not sure on which date to book appointment. 
    Input to this should not contain date"""

#     api_url = "https://889d-2402-a00-172-22e6-71e5-ba36-c2e7-3c81.ngrok-free.app/test/appointment/create"
    # api_url="https://495c-2402-a00-172-22e6-5ea8-c44e-fd0e-e8ed.ngrok-free.app/test/appointment/create"
    api_url="https://e182-52-73-21-156.ngrok-free.app/test/appointment/create"

    data_dict = {
    "company_id": company_id,
    "location_id": location_id,
    "lead": {
        "name": name,
        "phone": phone,
        "email": email
    },
    "vehicle": {
        "year": year,
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
    print(response.status_code)
    print("___json___")
    print(response.json)
    print("___text___")
    print(response.text)
#      response = requests.patch(api_url, json=data_dict)
   
    # Check the response status code
    if response.status_code == 200:
        print("Data stored successfully!")
        appointment_url=response.text
        return appointment_url
           
    else:
        print(f"Failed to store data. Status code: {response.status_code}")
        print(response.text)  # Print the response content for debugging



#####CONFORM APPOINTMENT######
class CustomerDataStore(BaseModel):
    name: str = Field(..., description="name of the customer")
    phone: str = Field(..., description="phone number of the customer")
    email: str = Field(..., description="email of the customer")
    make: str = Field(..., description="year of the car")
    model: str = Field(..., description="a single car or a multiple models of the car with comma separated")
    year:int=Field(..., description="year of the vehicle")
    company_id:int=Field(..., description="id of the company")
    location_id:int=Field(..., description="location id of the company")
    start_date:str=Field(..., description="date and time of appointment")
    appointment_timezone:str=Field(..., description="time zone")
    intent:str=Field(..., description="costumer intent")
    summary:str=Field(..., description="one line about summary of appointment,")
    description:str=Field(..., description="one line about description about visit,")

@tool
def confirm_appointment(name: str,phone: str,email: str ,make: str,model: str,year:int,
                           company_id:int,location_id:int,start_date:str,appointment_timezone:str,
                           intent:str,summary:str,description:str) -> dict:



    """Use this tool to confirm appointment for the given date and time"""
#     print(data)
    
    # Your API endpoint for storing appointment data
#     api_url = "https://889d-2402-a00-172-22e6-71e5-ba36-c2e7-3c81.ngrok-free.app/test/appointment/create"
    api_url="https://webapp-api-green.prod.funnelai.com/test/appointment/create"

    data_dict = {
    "company_id": company_id,
    "location_id": location_id,
    "lead": {
        "name": name,
        "phone": phone,
        "email": email
    },
    "vehicle": {
        "year": year,
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
AIRTABLE_TABLE_NAME = "new_apis"


st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/buravelliprasad/turbo_6_tools/blob/main/car_desription_new.csv) to get a sense for what questions you can ask.")

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


template = """You are an costumer care support exectutive based on your performance you will get bonus and incentives 
so follow instructions strictly and respond in Personable, Persuvasive, creative, engaging, witty and professional.
The name of the costumer is {name} and the dealership name is {dealership_name}. 
Do not start with appointment related questions.
To ensure a consistent and effective response, please adhere to the following guidelines:
Inventory related Questions: 
use "details_of_car" tool that extracts comprehensive information about cars in our inventory and also checks availability.

Avoid combining multiple questions like given below example1
example1:  "Are you interested in a new or used car or specific make or model in mind Or any specific features 
like towing capacity, off-road capability?"

When customers inquire about specific car features like towing, off-road capability, mileage, pickup trucks, 
or family cars if there is mention of new or used ask them are they interested in a new or used vehicle.


**DO NOT DISCLOSE PRICE**  
Do not disclose or ask the costumer if he likes to know the selling price of a car,
disclose selling price only when the customer explicitly requests it.

When utilizing the "details_of_car" tool, please respond with a list of cars, excluding square brackets. 
For each car, include the make, year, model and trim.
Additionally, strictly provide their car_details_link and also provide car_images_link.

Partition the list with new cars listed first, followed by a separate section for used cars.

When using the 'details_of_car' tool to provide car information, adhere to these guidelines 
to ensure concise and non-redundant responses:

1. Prioritize Uniqueness:

Consider cars as unique entities when they differ in any of the following core features:
Model
Make
Year
Trim
Exterior color
Interior color
New/used status
Cars sharing identical values for all of these features are considered similar.

2. Avoid Similar Car Duplication:

Display only one instance of a car if other cars with identical core features are present within the dataset.
This ensures concise responses that highlight distinct vehicles without redundancy.
Example:
If two cars have the same make, model, year, trim, exterior color, interior color, and new/used status 
display only one of them in the response.

If the output from the 'details_of_car' tool yields multiple results, and each car model is distinct,
kindly inquire with the customer to confirm their preferred choice.
This will ensure clarity regarding the specific model that piques the customer's interest.


After providing car details, kindly ask the customer if he is interested for test drive.

you: Are you interested in test drive?
customer:yes
you:check for contact number in record, politely inquire about it.
customer: phone number given
you: preffered appointment date?

For certain date and time:

{details}. 
Given this information, please use the "get_appointment_details" tool to check for the availability of the 
requested appointment date by the customer. For example If the costumer requested date is tomorrow, 
you know todays date and day find tomorrows date and use "get_appointment_details" tool. Use todays date only when costumer requests 
for todays appointment else use todays date for finding requested date.
If the desired date is available, proceed to book the appointment using the "confirm_appointment" tool.

In case the requested date is unavailable, kindly suggest alternative slots close to the customer's preference

Ask the customer for their preferred date and time for a test drive.
{details} using these details find appointment date requested by costumer 
and use the "get_appointment_details" tool to check availability.
If available, book the appointment with "confirm_appointment".
If unavailable, suggest alternative slots close to their preference.

For uncertain date and time:

Use the "create_appointment_link" tool to generate a booking link.
Share the "book now" link with the customer.
After fixing appointment or providing the appointment link Follow up question should be from Post-Appointment.
Post-Appointment follow this steps strictly. only one question at a time no multiple questions in single response :

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
    

Encourage Dealership Visit: Our aim to encourage customers to explore our dealership for test drives. 
Once essential details about the car, including make, model, color, and core features 
are provided extend a cordial invitation to schedule a test drive.

Business details: Enquiry regarding google maps location and address of the store and contact details use 
search_business_details tool.

company details:
compant id is 39, location id is 1 and timezone is America/New_York

Strictly Keep responses concise, not exceeding two sentences or 100 words and answers should be interactive.
Respond in a polite US english.

**strictly answer only from the  content provided to you dont makeup answers.**"""
details= "Today's date is "+ todays_date +" in mm-dd-yyyy format and its "+day_of_the_week+"."
name = st.session_state.user_name
dealership_name="Gosch Chevrolet"
input_template = template.format(details=details,name=name,dealership_name=dealership_name)
system_message = SystemMessage(content=input_template)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)
tools = [tool1,tool3,check_appointment_availability,confirm_appointment,create_appointment_link]
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
if 'agent_executor' not in st.session_state:
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_source_documents=True,
        return_generated_question=True)
    st.session_state.agent_executor = agent_executor
else:
    agent_executor = st.session_state.agent_executor
    
chat_history=[]
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

def convert_text_to_html_images(text):
    # Pattern to match the specific format
    pattern = r"image_url:([^,]+), car_details_url:([^,\s]+)"
    
    # Function to replace each match with an HTML string
    def replace_with_html(match):
        image_url = match.group(1).strip()
        car_details_url = match.group(2).strip()
        return f'<a href="{car_details_url}" target="_blank"><img src="{image_url}" alt="Car Image" style="width:100px;height:auto;"/></a>'
    
    # Replace all occurrences in the text
    html_text = re.sub(pattern, replace_with_html, text)
    return html_text
    
# def extract_inventory_page_url(text):
#     # Regular expression to match the inventory page URL in the provided text
#     pattern = r'\[Details\]\(([^)]+)\)'
    
#     # Search for the pattern in the text
#     match = re.search(pattern, text)
    
#     # If a match is found, return the extracted URL; otherwise, return None
#     return match.group(1) if match else None
def extract_inventory_page_url(text):
    # Regular expression to match the inventory page URL in the provided text
    pattern = r'\[(Details|Car Details|View Details)\]\(([^)]+)\)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the extracted URL; otherwise, return None
    return match.group(2) if match else None



# def convert_links(text):
    
#     # Regular expression to match markdown format ![alt text](URL) or [link text](URL)
#     pattern = r'!?\[([^\]]+)\]\(([^)]+)\)'

#     # Function to replace each match
#     def replace_with_tag(match):
#         prefix = match.group(0)[0]  # Check if it's an image or a link
#         alt_or_text = match.group(1)
#         url = match.group(2)
#         # Check for common image file extensions
#         if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
#             return f'<a href="{url}"><img src="{url}" alt="{alt_or_text}" style="width: 100px; height: auto;"/></a>'

#         else:
#             return f'<a href="{url}">{alt_or_text}</a>'

#     # Replace all occurrences
#     html_text = re.sub(pattern, replace_with_tag, text)

#     return html_text 
# def convert_links(text):
#     # Regular expression to match markdown format ![alt text](URL) or [link text](URL)
#     pattern = r'!?\[([^\]]+)\]\(([^)]+)\)'

#     # Function to replace each match
#     def replace_with_tag(match):
#         prefix = match.group(0)[0]  # Check if it's an image or a link
#         alt_or_text = match.group(1)
#         url = match.group(2)
#         # Check for common image file extensions
#         if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
#             # Extracted inventory page URL
#             inventory_page_url = extract_inventory_page_url(text)
#             if inventory_page_url:
#                 return f'<a href="{inventory_page_url}" target="_blank"><img src="{url}" alt="{alt_or_text}" style="width: 100px; height: auto;"/></a>'
#             else:
#                 return f'<a href="{url}" target="_blank"><img src="{url}" alt="{alt_or_text}" style="width: 100px; height: auto;"/></a>'
#         else:
#             return f'<a href="{url}" target="_blank">{alt_or_text}</a>'

#     # Replace all occurrences
#     html_text = re.sub(pattern, replace_with_tag, text)

#     return html_text
def convert_links(text):
    # Regular expression to match markdown format ![alt text](URL) or [link text](URL)
    pattern = r'!?\[([^\]]+)\]\(([^)]+)\)'

    # Function to replace each match
    def replace_with_tag(match):
        prefix = match.group(0)[0]  # Check if it's an image or a link
        alt_or_text = match.group(1)
        url = match.group(2)
        # Check for common image file extensions
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            # Extracted inventory page URL
            inventory_page_url = extract_inventory_page_url(text)
            if inventory_page_url:
                return f'<a href="{inventory_page_url}" target="_blank"><img src="{url}" alt="{alt_or_text}" style="width: 100px; height: auto;"/></a>'
            else:
                return f'<a href="{url}" target="_blank"><img src="{url}" alt="{alt_or_text}" style="width: 100px; height: auto;"/></a>'
        else:
            return f'<a href="{url}" target="_blank">{alt_or_text}</a>'

    # Replace all occurrences
    html_text = re.sub(pattern, lambda match: replace_with_tag(match), text)

    return html_text
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
                # st.markdown(convert_links(answer))
                st.markdown(
                    f'<div style="background-color: black; color: white; border-radius: 10px; padding: 10px; width: 85%;'
                    f' border-top-right-radius: 10px; border-bottom-right-radius: 10px;'
                    f' border-top-left-radius: 0; border-bottom-left-radius: 0; box-shadow: 2px 2px 5px #888888;">'
                    f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{convert_links(answer)}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        # if st.session_state.user_name:
        #     try:
        #         save_chat_to_airtable(st.session_state.user_name, user_input, output)
        #     except Exception as e:
        #         st.error(f"An error occurred: {e}")
