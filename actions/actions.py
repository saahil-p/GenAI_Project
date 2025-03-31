# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import os
import torch
import hashlib
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
    print("Using new langchain_chroma package.")
except ImportError:
    from langchain.vectorstores import Chroma
    print("langchain_chroma not installed. Falling back to legacy import.")

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device("mps")
    print("Using MPS for acceleration.")
else:
    device = torch.device("cpu")
    print("MPS not available. Falling back to CPU.")

TXT_DIRECTORY = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG"
CHROMA_PATH = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG/chroma_db"
MODEL_ID = "google/flan-t5-small"

def get_embedding_fn():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_chroma_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_fn())

def get_retriever(db):
    return db.as_retriever(search_kwargs={"k": 3})

def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16
    )
    llm_pipeline = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=250, min_new_tokens=10, do_sample=False
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)

def setup_qa_pipeline(db):
    retriever = get_retriever(db)
    llm = get_llm()
    prompt_template = """<|system|>
You are an expert assistant specialized in oil well extraction. Your task is to provide accurate, concise information based ONLY on the context provided.
If the context doesn't contain enough information to answer the question, you must say "I don't have enough information to answer this question."
Do not make up information or rely on prior knowledge not present in the context.
Respond in a clear, formal, and professional manner. 
<|user|>
Context information:
{context}

Based on this context, answer the following question: {question}
<|assistant|>
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

def run_rag_query(query_text: str) -> str:
    db_instance = get_chroma_db()
    qa_pipeline = setup_qa_pipeline(db_instance)
    response = qa_pipeline.invoke({"query": query_text})
    return response['result']

def run_sensor_query(query_text: str, sensor_name: str) -> str:
    db_instance = get_chroma_db()
    qa_pipeline = setup_qa_pipeline(db_instance)
    
    # Create context from sensor description
    sensor_context = f"""
    The sensor {sensor_name} is {SENSOR_DESCRIPTIONS.get(sensor_name, 'not found in the system')}.
    This sensor is part of an oil well monitoring system.
    """
    
    # Modify the prompt to include sensor context
    prompt_template = """<|system|>
You are an expert assistant specialized in oil well sensors and monitoring. Your task is to provide accurate, concise information based on the context provided.
If the context doesn't contain enough information to answer the question, you must say "I don't have enough information to answer this question."
Do not make up information or rely on prior knowledge not present in the context.
Respond in a clear, formal, and professional manner. 
<|user|>
Context information:
{context}

Based on this context, answer the following question: {question}
<|assistant|>
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create a new chain with the modified prompt
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=db_instance.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Combine the sensor context with the query
    combined_query = f"{sensor_context}\n\nQuestion: {query_text}"
    response = chain.invoke({"query": combined_query})
    return response['result']

def run_well_status_query(query_text: str, well_number: str) -> str:
    try:
        print(f"Starting well status query for well {well_number}")
        db_instance = get_chroma_db()
        print("Got chroma DB instance")
        
        # Create context from well number
        well_context = f"""
        Well {well_number} is one of the monitored oil wells in the production field.
        This well is equipped with various sensors for monitoring pressure, temperature, flow rates, and valve states.
        Well numbers are used to identify specific extraction points in the oil field.
        """
        
        # Modify the prompt to include well context
        prompt_template = """<|system|>
You are an expert assistant specialized in oil well monitoring and operations. Your task is to provide accurate, concise information based on the context provided.
If the context doesn't contain enough information to answer the question, you must say "I don't have enough information to answer this question, but I can provide general information about oil well operations."
Do not make up specific operational data, but you can explain how well monitoring systems work in general.
Respond in a clear, formal, and professional manner.
<|user|>
Context information:
{context}

Based on this context, answer the following question: {question}
<|assistant|>
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        print("Created prompt template")
        
        # Create a new chain with the modified prompt
        chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=db_instance.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("Created chain")
        
        # Combine the well context with the query
        combined_query = f"{well_context}\n\nQuestion: {query_text}"
        print("Sending query to chain")
        
        # Invoke the chain with properly formatted inputs
        response = chain.invoke({"query": combined_query})
        print("Got response from chain")
        return response['result']
    except Exception as e:
        print(f"Error in run_well_status_query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}"

class ActionRAGQuery(Action):
    def name(self) -> Text:
        return "action_rag_query"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_query = tracker.latest_message.get("text")
        print(user_query)
        if user_query:
            response = run_rag_query(user_query)
            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(text="I couldn't understand your question. Please try again.")
        return []

# Dictionary mapping sensor names to their descriptions
SENSOR_DESCRIPTIONS = {
    "timestamp": "Instant at which observation was generated",
    "ABER-CKGL": "Opening of the GLCK (gas lift choke) [%%]",
    "ABER-CKP": "Opening of the PCK (production choke) [%%]",
    "ESTADO-DHSV": "State of the DHSV (downhole safety valve) [0, 0.5, or 1]",
    "ESTADO-M1": "State of the PMV (production master valve) [0, 0.5, or 1]",
    "ESTADO-M2": "State of the AMV (annulus master valve) [0, 0.5, or 1]",
    "ESTADO-PXO": "State of the PXO (pig-crossover) valve [0, 0.5, or 1]",
    "ESTADO-SDV-GL": "State of the gas lift SDV (shutdown valve) [0, 0.5, or 1]",
    "ESTADO-SDV-P": "State of the production SDV (shutdown valve) [0, 0.5, or 1]",
    "ESTADO-W1": "State of the PWV (production wing valve) [0, 0.5, or 1]",
    "ESTADO-W2": "State of the AWV (annulus wing valve) [0, 0.5, or 1]",
    "ESTADO-XO": "State of the XO (crossover) valve [0, 0.5, or 1]",
    "P-ANULAR": "Pressure in the well annulus [Pa]",
    "P-JUS-BS": "Downstream pressure of the SP (service pump) [Pa]",
    "P-JUS-CKGL": "Downstream pressure of the GLCK (gas lift choke) [Pa]",
    "P-JUS-CKP": "Downstream pressure of the PCK (production choke) [Pa]",
    "P-MON-CKGL": "Upstream pressure of the GLCK (gas lift choke) [Pa]",
    "P-MON-CKP": "Upstream pressure of the PCK (production choke) [Pa]",
    "P-MON-SDV-P": "Upstream pressure of the production SDV (shutdown valve) [Pa]",
    "P-PDG": "Pressure at the PDG (permanent downhole gauge) [Pa]",
    "PT-P": "Downstream pressure of the PWV (production wing valve) in the production tube [Pa]",
    "P-TPT": "Pressure at the TPT (temperature and pressure transducer) [Pa]",
    "QBS": "Flow rate at the SP (service pump) [m3/s]",
    "QGL": "Gas lift flow rate [m3/s]",
    "T-JUS-CKP": "Downstream temperature of the PCK (production choke) [oC]",
    "T-MON-CKP": "Upstream temperature of the PCK (production choke) [oC]",
    "T-PDG": "Temperature at the PDG (permanent downhole gauge) [oC]",
    "T-TPT": "Temperature at the TPT (temperature and pressure transducer) [oC]"
}

class ActionSensorQuery(Action):
    def name(self) -> Text:
        return "action_sensor_query"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the latest message and entities
        message = tracker.latest_message.get("text", "")
        entities = tracker.latest_message.get("entities", [])
        
        # Find the sensor entity
        sensor_entity = next((entity for entity in entities if entity.get("entity") == "sensor"), None)
        
        if sensor_entity:
            sensor_name = sensor_entity.get("value")
            if sensor_name in SENSOR_DESCRIPTIONS:
                # Use RAG with sensor context
                response = run_sensor_query(message, sensor_name)
            else:
                response = f"I don't have information about the sensor '{sensor_name}'."
        else:
            response = "I couldn't identify which sensor you're asking about. Please specify the sensor name."
            
        dispatcher.utter_message(text=response)
        return []

class ActionWellStatus(Action):
    def name(self) -> Text:
        return "action_well_status"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            print("Starting ActionWellStatus")
            # Get the latest message
            user_message = tracker.latest_message.get("text", "")
            print(f"User message: {user_message}")
            
            # Get the well number entity from the message
            well_number = tracker.get_slot("well_number")
            
            if not well_number:
                # Check if the entity is in the latest message but not yet set as a slot
                entities = tracker.latest_message.get("entities", [])
                well_entity = next((entity for entity in entities if entity.get("entity") == "well_number"), None)
                if well_entity:
                    well_number = well_entity.get("value")
            
            print(f"Extracted well number: {well_number}")
            
            if well_number:
                # These are the possible operational statuses for wells
                operational_statuses = [
                    "NORMAL", 
                    "ABRUPT_INCREASE_OF_BSW", 
                    "SPURIOUS_CLOSURE_OF_DHSV", 
                    "SEVERE_SLUGGING", 
                    "FLOW_INSTABILITY", 
                    "RAPID_PRODUCTIVITY_LOSS", 
                    "QUICK_RESTRICTION_IN_PCK", 
                    "SCALING_IN_PCK", 
                    "HYDRATE_IN_PRODUCTION_LINE", 
                    "HYDRATE_IN_SERVICE_LINE"
                ]
                
                # Simulate DNN prediction for well operational status
                import random
                random.seed(int(well_number))
                predicted_status = random.choice(operational_statuses)
                print(f"Predicted status: {predicted_status}")
                
                # Provide descriptions for each status
                status_descriptions = {
                    "NORMAL": "Well is operating within expected parameters. All sensors show normal readings.",
                    "ABRUPT_INCREASE_OF_BSW": "Sudden increase in Basic Sediment and Water (BSW) detected, indicating possible water breakthrough.",
                    "SPURIOUS_CLOSURE_OF_DHSV": "Unexpected closure of the Downhole Safety Valve detected, restricting flow.",
                    "SEVERE_SLUGGING": "Pronounced alternating liquid and gas flow detected, causing pressure fluctuations.",
                    "FLOW_INSTABILITY": "Irregular flow patterns detected, leading to production inefficiencies.",
                    "RAPID_PRODUCTIVITY_LOSS": "Significant decrease in production rate observed, indicating possible reservoir issues.",
                    "QUICK_RESTRICTION_IN_PCK": "Sudden reduction in flow through the Production Choke, indicating possible obstruction.",
                    "SCALING_IN_PCK": "Mineral scale buildup detected in the Production Choke, restricting flow capacity.",
                    "HYDRATE_IN_PRODUCTION_LINE": "Formation of gas hydrates detected in the production line, potentially leading to blockage.",
                    "HYDRATE_IN_SERVICE_LINE": "Gas hydrate formation detected in the service line, affecting well support systems."
                }
                
                # Get the description for the predicted status
                status_description = status_descriptions.get(predicted_status, "Status description not available.")
                
                # Use a question that focuses on explaining the predicted status
                rag_query = f"What does {predicted_status} mean for Well {well_number}? What actions should be taken for this condition? Explain the implications for production and safety."
                
                # Get RAG response
                print("Calling run_well_status_query")
                response = run_well_status_query(rag_query, well_number)
                print("Got response from run_well_status_query")
                
                # Prepend the predicted status to the RAG response for clarity
                final_response = f"**Current Status of Well {well_number}**: {predicted_status}\n\n**Description**: {status_description}\n\n**Analysis**: {response}"
                
            else:
                # Generic response for wells in general
                final_response = "I don't have information about specific wells at the moment. To get detailed well status, please specify the well number."
            
            print("Sending response to user")
            dispatcher.utter_message(text=final_response)
            return []
            
        except Exception as e:
            print(f"Error in ActionWellStatus: {str(e)}")
            dispatcher.utter_message(text=f"I encountered an error while processing your request: {str(e)}")
            return []
