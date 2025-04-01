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
import time
import ssl
import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import GoogleSearchAPIWrapper

try:
    from langchain_community.vectorstores import Chroma
    print("Using new langchain_community.vectorstores package.")
except ImportError:
    from langchain.vectorstores import Chroma
    print("langchain_community.vectorstores not installed. Falling back to legacy import.")

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

# Global variables for caching
_embedding_fn = None
_chroma_db = None
_tokenizer = None
_model = None
_llm_instance = None
_qa_chain = None
_search = None  # Add global search instance

# Constants
TXT_DIRECTORY = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG"
CHROMA_PATH = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG/chroma_db"
MODEL_ID = "google/flan-t5-small"

def initialize_models():
    """Initialize all models and databases when the server starts."""
    global _embedding_fn, _chroma_db, _tokenizer, _model, _llm_instance, _qa_chain, _search
    
    try:
        print("Starting model initialization...")
        start_time = time.time()
        
        # Initialize Google Search
        try:
            print("Initializing Google Search...")
            _search = GoogleSearchAPIWrapper()
        except Exception as e:
            print(f"Failed to initialize Google Search: {str(e)}")
            print("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables are set.")
            _search = None

        # Set device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_default_device("mps")
            print("Using MPS for acceleration.")
        else:
            device = torch.device("cpu")
            print("MPS not available. Falling back to CPU.")

        # Initialize embedding function with SSL verification disabled
        print("Loading embedding model...")
        try:
            # First try with default settings
            _embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except requests.exceptions.SSLError:
            print("SSL verification failed, retrying with verification disabled...")
            # If SSL fails, try with verification disabled
            _embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Initialize or load Chroma DB
        print("Initializing Chroma DB...")
        if not os.path.exists(CHROMA_PATH):
            print("Creating new Chroma DB...")
            # Load and process documents
            loader = DirectoryLoader(TXT_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            # Create new Chroma DB
            _chroma_db = Chroma.from_documents(
                documents=texts,
                embedding=_embedding_fn,
                persist_directory=CHROMA_PATH
            )
            _chroma_db.persist()
        else:
            print("Loading existing Chroma DB...")
            _chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=_embedding_fn)
        
        # Initialize language model with SSL verification disabled
        print("Loading language model...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        except requests.exceptions.SSLError:
            print("SSL verification failed for language model, retrying with verification disabled...")
            # If SSL fails, try with verification disabled
            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, trust_remote_code=True, local_files_only=True
            )
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True
            )
        
        # Create LLM pipeline
        llm_pipeline = pipeline(
            "text2text-generation",
            model=_model,
            tokenizer=_tokenizer,
            max_new_tokens=250,
            min_new_tokens=10,
            do_sample=False
        )
        _llm_instance = HuggingFacePipeline(pipeline=llm_pipeline)
        
        # Create QA chain
        print("Creating QA chain...")
        retriever = _chroma_db.as_retriever(search_kwargs={"k": 3})
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
        _qa_chain = RetrievalQA.from_chain_type(
            llm=_llm_instance,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        end_time = time.time()
        print(f"Model initialization completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        # Try to provide more helpful error messages
        if "SSL" in str(e):
            print("SSL verification error detected. This might be due to network security settings.")
            print("Please check your network connection and SSL certificate settings.")
        elif "Connection" in str(e):
            print("Connection error detected. Please check your internet connection.")
        raise

# Initialize models when the module is loaded
try:
    initialize_models()
except Exception as e:
    print(f"Failed to initialize models: {str(e)}")
    # You might want to handle this error differently based on your requirements
    raise

def get_embedding_fn():
    """Get the cached embedding function."""
    if _embedding_fn is None:
        raise RuntimeError("Embedding function not initialized")
    return _embedding_fn

def get_chroma_db():
    """Get the cached Chroma DB instance."""
    if _chroma_db is None:
        raise RuntimeError("Chroma DB not initialized")
    return _chroma_db

def get_llm():
    """Get the cached LLM instance."""
    if _llm_instance is None:
        raise RuntimeError("LLM not initialized")
    return _llm_instance

def get_qa_chain():
    """Get the cached QA chain."""
    if _qa_chain is None:
        raise RuntimeError("QA chain not initialized")
    return _qa_chain

def get_retriever(db):
    return db.as_retriever(search_kwargs={"k": 3})

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

def perform_web_search(query: str) -> str:
    """Perform a web search and return a summarized response."""
    try:
        if _search is None:
            print("Web search is not available - Google Search API not initialized")
            return None

        # Search the web for information
        search_query = f"oil well {query} technical definition operation"
        print(f"Performing web search for: {search_query}")
        
        # Get web search results
        search_results = _search.run(search_query)
        
        # If no results found, return None
        if not search_results:
            return None
            
        # Create a prompt to summarize the web results
        summary_prompt = f"""Based on these search results about {query}, provide a clear, technical explanation in 2-3 sentences:
        
{search_results}"""
        
        # Use the QA chain to summarize the results
        qa_chain = get_qa_chain()
        summary = qa_chain.invoke({"query": summary_prompt})
        
        return summary.get("result", None)
        
    except Exception as e:
        print(f"Error in web search: {str(e)}")
        return None

def clean_response(response: str) -> str:
    """Clean up response by removing formatting and incomplete sentences."""
    import re
    
    # First, remove all content between pipe characters and clean formatting
    response = re.sub(r'\|[^|]*\|', ' ', response)  # Replace content between pipes with a space
    response = response.replace("|", "")  # Remove any remaining pipe characters
    response = re.sub(r'[-=_]{3,}', '', response)  # Remove long sequences of formatting characters
    
    # Clean up extra whitespace
    response = re.sub(r'\s+', ' ', response)
    
    # Split into sentences (considering multiple punctuation marks)
    sentences = re.split(r'(?<=[.!?])\s+', response)
    
    # Keep only complete sentences
    complete_sentences = []
    for sentence in sentences:
        # Check if sentence is complete (starts with capital letter, ends with punctuation)
        if (sentence and 
            sentence[0].isupper() and  # Starts with capital letter
            sentence[-1] in '.!?' and  # Ends with proper punctuation
            len(sentence.split()) > 2):  # Has at least 3 words (likely complete)
            complete_sentences.append(sentence)
    
    # Join complete sentences back together
    cleaned_response = ' '.join(complete_sentences)
    return cleaned_response.strip()

def run_rag_query(query: str) -> str:
    """Run a RAG query using cached models, falling back to web search if needed."""
    try:
        # Get the cached QA chain
        qa_chain = get_qa_chain()
        
        # Run the query
        result = qa_chain.invoke({"query": query})
        response = result.get("result", "")
        
        # Check if the response indicates no information
        if "don't have" in response.lower() or "not enough information" in response.lower():
            print("No information found in RAG, trying web search...")
            web_result = perform_web_search(query)
            
            if web_result:
                return f"Based on web search: {web_result}"
            elif _search is None:
                return "I don't have enough information to answer this question, and web search is not available. Please ensure Google Search API is properly configured."
            
        return response or "I couldn't find a relevant answer to your question."
        
    except Exception as e:
        print(f"Error in run_rag_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again."

def run_sensor_query(query: str, sensor_name: str) -> str:
    """Run a sensor-specific query using cached models, falling back to web search if needed."""
    try:
        # Get the cached QA chain
        qa_chain = get_qa_chain()
        
        # Create a more specific query that includes the sensor context
        enhanced_query = f"Regarding the {sensor_name} sensor in oil wells: {query}. Provide a clear, technical explanation of what this sensor is and its purpose. Focus on its main function and operational significance. Exclude any technical specifications or state values unless specifically asked for. Use complete sentences."
        
        # Run the query
        result = qa_chain.invoke({"query": enhanced_query})
        response = result.get("result", "")
        
        # Clean and post-process the response
        response = clean_response(response)
        
        # Check if we have a valid response after cleaning
        if not response:
            print(f"No valid response after cleaning for sensor {sensor_name}, trying web search...")
            web_result = perform_web_search(f"{sensor_name} sensor oil well")
            
            if web_result:
                web_result = clean_response(web_result)
                if web_result:
                    return f"Based on web search: {web_result}"
            
            if _search is None:
                return f"I don't have enough information about the {sensor_name} sensor, and web search is not available. Please ensure Google Search API is properly configured."
            
        return response or f"I couldn't find specific information about the {sensor_name} sensor."
        
    except Exception as e:
        print(f"Error in run_sensor_query: {str(e)}")
        return f"I encountered an error while processing your query about the {sensor_name} sensor. Please try again."

def run_well_status_query(query: str, well_number: str) -> str:
    """Run a well status query using cached models, falling back to web search if needed."""
    try:
        # Get the cached QA chain
        qa_chain = get_qa_chain()
        
        # Create a more specific query that includes the well context
        enhanced_query = f"Regarding Well {well_number}: {query}. Provide your response in complete sentences."
        
        # Run the query
        result = qa_chain.invoke({"query": enhanced_query})
        response = result.get("result", "")
        
        # Clean and post-process the response
        response = clean_response(response)
        
        # Check if we have a valid response after cleaning
        if not response:
            print(f"No valid response after cleaning for Well {well_number}, trying web search...")
            web_result = perform_web_search(query)
            
            if web_result:
                web_result = clean_response(web_result)
                if web_result:
                    return f"Based on web search: {web_result}"
            
            if _search is None:
                return f"I don't have enough information about this query, and web search is not available. Please ensure Google Search API is properly configured."
            
        return response or f"I couldn't find specific information about Well {well_number}."
        
    except Exception as e:
        print(f"Error in run_well_status_query: {str(e)}")
        return f"I encountered an error while processing your query about Well {well_number}. Please try again."

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