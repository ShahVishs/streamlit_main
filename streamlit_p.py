from pydantic import BaseModel, Field
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
import datetime
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
from openai import OpenAI
from langchain.document_loaders import JSONLoader
from langchain_core.tracers.langchain_v1 import LangChainTracerV1
import re
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

file_1 = r'inventory_goush_cleaned_new.csv'

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

file_3 = r'csvjson.json'
# loader_3 = JSONLoader(file_path=file_3,jq_schema='.messages[].content',text_content=False)
# data_3 = loader_3.load()
loader_3 = JSONLoader(file_path=file_3, jq_schema='.', text_content=False)
data_3 = loader_3.load()
vectordb_3 = FAISS.from_documents(data_3, embeddings)
retriever_4 = vectordb_3.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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

tool4 = create_retriever_tool(
    retriever_4, 
     "image_details",
     "Use to search for vehicle information and images based on make and model."
)

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
template = """You are an costumer care support exectutive respond in 
Personable, Humorous, emotional intelligent, creative, witty and engaging.
The name of the costumer is {name} and the dealership name is {dealership_name} and 
never start with appointment related questions.
To ensure a consistent and effective response, please adhere to the following guidelines:
Use "Availability_check" for checking availability of a specific make or model of the car and 
also for getting full list of available makes and models in the inventory. 
Use "car_details" tool that extracts comprehensive information about cars in the inventory.
This includes details like trim, price, color, and cost. To optimize the search process, 
ensure the system is aware of the car model and whether the customer is interested in new or used cars.
In cases where specific details are not included in the initial inquiry, 
initiate a proactive approach by requesting the missing information. 
To streamline the process, ask only one question at a time until all necessary details are obtained.
This ensures a more efficient and accurate retrieval of car information.
If customer inquires about car with features like towing, off-road capability,
good mileage, or pickup trucks, in this case no need to ask about make and model of the car inquire 
whether they are interested in a new or used vehicle. After knowing car feature and new or old car preference 
use the "details_of_car" tool to answer.
Use "Availability_check" for checking car availability and "car_details" for car information.
Do not disclose or ask the costumer if he likes to know the selling price of a car,
disclose selling price only when the customer explicitly requests it use "details_of_car" tool.
Here's a suggested response format while providing car details:
"We have several models available. Here are a few options:"
If the customer's query matches a car model, respond with a list of car without square brackets, 
including the make, year, model, and trim, and provide their respective links in the answer.

checking Appointments Avaliability: If inquiry lacks specific details like day, date or time kindly engage by 
asking for these specifics.
{details} Use these details and find appointment date from the users input and check for appointment availabity 
using "appointment_scheduling" tool for that specific day or date and time.
use pandas dataframe `df` in Python.
This is the result of running `df.head().to_markdown()`. 
<df>
{dhead}
</df>
You are not meant to use only these rows to answer questions - they are meant as a way of telling you\nabout the 
shape and schema of the dataframe.
you can run intermediate queries to do exporatory data analysis to give you more information as needed. 
If the requested date and time for the appointment are unavailable,
suggest alternative times close to the customer's preference.

Additionally, provide this link'[click here](https://app.engagedai.io/engagements/appointment)'it will take them to a URL where they
can schedule or reschedule their appointment themselves. 
Appointment Scheduling:

After scheduling an appointment, initiate the conversation to get tradein car and personal details.
**Car Trade-In Inquiry and personal details:**

1. Ask the customer if they have a car for trade-in.

    - User: [Response]

2. If the user responds with "Yes" to trade-in, ask for the VIN (Vehicle Identification Number).

    - User: [Response]

3. If the user responds with "No" to the VIN, ask for the make, model, and year of the car.

    - User: [Response]

**Price Expectation:**

4. Once you have the trade-in car details, ask the customer about their expected price for the trade-in.

    - User: [Response]

**Personal Information:**

5. Finally, ask for the customer's personal details.

    - User: [Response]
    - Name:
    - Contact Number:
    - Email Address:

**Vehicle Image:**

Show image of a specific vehicle that user ask, provide the make and model, and I'll fetch the corresponding image for you.
Use the "image_details" tool for this purpose.

Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
receive product briefings from our team. After providing essential information on the car's make, model,
color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
for a comprehensive product overview by our experts.
Business details: Enquiry regarding google maps location of the store, address of the store, working days and working hours 
and contact details use search_business_details tool to get information.


Keep responses concise, not exceeding two sentences and answers should be interactive.
Respond in a polite US english.
answer only from the provided content dont makeup answers.

"""
details= "Today's current date is "+ todays_date +" today's weekday is "+day_of_the_week+"."

name = st.session_state.user_name
dealership_name="Pine belt cars"


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
df = pd.read_csv("appointment_new.csv")

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


input_template = template.format(dhead=df.iloc[:3, :5].to_markdown(),details=details,name=name,dealership_name=dealership_name)
system_message = SystemMessage(content=input_template)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)


repl = PythonAstREPLTool(locals={"df": df}, name="appointment_scheduling",
        description="Use to check on available appointment times for a given date and time. The input to this tool should be a string in this format mm/dd/yy.This tool will reply with available times for the specified date in 12 hour time, for example: 15:00 and are the same")

tools = [tool1, repl, tool2, tool3, tool4]
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

client = OpenAI()

def load_car_data(file_path):
    with open(file_path, 'r') as file:
        car_data = json.load(file)
    return car_data

car_data = load_car_data(r"csvjson.json")

def get_car_information(make, model):
    """Get information about a car based on make and model."""
    matching_cars = [car for car in car_data if car["Make"].lower() == make.lower() and car["Model"].lower() == model.lower()]

    if matching_cars:
        return json.dumps(matching_cars)
    else:
        return json.dumps({"error": "Car not found"})

def display_car_info_with_link(car_info_list, link_url, size=(300, 300)):
    try:
        for car_info in car_info_list:
            image_links = car_info.get("website Link for images")
            vin_number = car_info.get("Vin")  
            year = car_info.get("Year")
            make = car_info.get("Make")
            model = car_info.get("Model")

            for image_link in re.findall(r'https://[^ ,]+', image_links):
                response = requests.get(image_link)
                response.raise_for_status()
                image_data = Image.open(BytesIO(response.content))
                resized_image = image_data.resize(size)

                vin_number_from_url = re.search(r'/inventory/([^/]+)/', image_link)
                vin_number_from_info = vin_number or (vin_number_from_url.group(1) if vin_number_from_url else None)
                link_with_vin = f'{link_url}/{vin_number_from_info}/' if vin_number_from_info else link_url

                # Print or log image details for debugging
                print(f"Image Details - VIN: {vin_number_from_info}, Link: {link_with_vin}")

                # Display image in Streamlit
                st.image(resized_image, caption=f"{year} {make} {model}")

    except Exception as e:
        print(f"Error displaying car information: {e}")


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def run_conversation(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_car_information",
                "description": "Get information about a car",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "make": {"type": "string", "description": "The car make"},
                        "model": {"type": "string", "description": "The car model"}
                    },
                    "required": ["make", "model"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_car_information": get_car_information,
        }

        messages.append(response_message)  
        car_info_list = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                make=function_args.get("make"),
                model=function_args.get("model"),
            )

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  

            current_car_info_list = json.loads(function_response)
            if current_car_info_list:
                car_info_list.extend(current_car_info_list)

        # If no tool calls match, return an empty list
        return car_info_list

    return []
        
# def conversational_chat(user_input, user_name):
#     input_with_username = f"{user_name}: {user_input}"
#     result = agent_executor({"input": input_with_username})
#     output = result["output"]
#     st.session_state.chat_history.append((user_input, output))
    
#     return output
user_input = ""
car_info_list = run_conversation(user_input)

def conversational_chat(user_input, user_name):
    input_with_username = f"{user_name}: {user_input}"
    combined_input = f"{input_with_username} {car_info_list}"
    result = agent_executor({"input": combined_input})
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
        # output = conversational_chat(user_input, st.session_state.user_name, car_info_list)
        car_info_list = run_conversation(user_input)
    
        # Assuming the response from run_conversation contains car information
        link_url = "https://www.goschchevy.com/inventory/"
        display_car_info_with_link(car_info_list, link_url, size=(50, 50))
        
        # Display images in Streamlit
        for car_info in car_info_list:
            image_urls = car_info["website Link for images"].split(', ')
            
            for image_url in image_urls:
                st.image(image_url, caption=f"{car_info['Year']} {car_info['Make']} {car_info['Model']}")
            
            st.write(f"VIN: {car_info['Vin']}")
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
