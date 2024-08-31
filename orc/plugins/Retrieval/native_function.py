from shared.util import get_secret, get_aoai_config, extract_text_from_html, get_possitive_int_or_default
# from semantic_kernel.skill_definition import sk_function
from openai import AzureOpenAI
from semantic_kernel.functions import kernel_function
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
import logging
import os
import requests
import time
import sys
import json
import urllib
import pyodbc
from typing import Dict
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated
from azure.cognitiveservices.search.customsearch import CustomSearchClient
from msrest.authentication import CognitiveServicesCredentials
from sqlalchemy import create_engine
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding as LlamaAzureOpenAIEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine
from llama_index.core import SQLDatabase, Settings
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core import VectorStoreIndex
from azure.identity.aio import DefaultAzureCredential
import aiohttp
import asyncio

# Azure search Integration Settings
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")

TERM_SEARCH_APPROACH = 'term'
VECTOR_SEARCH_APPROACH = 'vector'
HYBRID_SEARCH_APPROACH = 'hybrid'
AZURE_SEARCH_USE_SEMANTIC = os.environ.get("AZURE_SEARCH_USE_SEMANTIC") or "false"
AZURE_SEARCH_APPROACH = os.environ.get("AZURE_SEARCH_APPROACH") or HYBRID_SEARCH_APPROACH

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION", "2023-11-01")
if AZURE_SEARCH_API_VERSION < '2023-10-01-Preview':  # query is using vectorQueries that requires at least 2023-10-01-Preview'.
    AZURE_SEARCH_API_VERSION = '2023-11-01'

AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K") or "3"

AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH") or "false"
AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH = True if AZURE_SEARCH_OYD_USE_SEMANTIC_SEARCH == "true" else False
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG") or "my-semantic-config"
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN") or "true"
AZURE_SEARCH_ENABLE_IN_DOMAIN = True if AZURE_SEARCH_ENABLE_IN_DOMAIN == "true" else False
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS") or "content"
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN") or "filepath"
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN") or "title"
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN") or "url"
# Bing search Integration Settings
BING_SEARCH_TOP_K = os.environ.get("BING_SEARCH_TOP_K") or "3"
BING_CUSTOM_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/custom/search?"
BING_SEARCH_MAX_TOKENS = os.environ.get("BING_SEARCH_MAX_TOKENS") or "1000"
AZURE_SEARCH_TRIMMING = os.environ.get("AZURE_SEARCH_TRIMMING") or False
# DB Integration Settings
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_TEMPERATURE = os.getenv("AZURE_OPENAI_TEMPERATURE", "0.17")
AZURE_OPENAI_APIVERSION = os.environ.get("AZURE_OPENAI_APIVERSION")
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_APIVERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_APIVERSION")
DB_SERVER = os.environ.get("DB_SERVER")
DB_DATABASE = os.environ.get("DB_DATABASE")
DB_USERNAME = os.environ.get("DB_USERNAME")
DB_TOP_K = os.environ.get("DB_TOP_K")
DB_MAX_TOKENS = os.environ.get("DB_MAX_TOKENS")
TOP_K_DEFAULT=3
MAX_TOKENS_DEFAULT=1000
DB_TYPE = os.environ.get("DB_TYPE")
# Set up logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
APIM_ENABLED = os.environ.get('APIM_ENABLED', 'false').lower() == 'true'
APIM_BING_CUSTOM_SEARCH_URL=os.environ.get('APIM_BING_CUSTOM_SEARCH_URL',"")+"/search?"
APIM_AZURE_SEARCH_URL=os.environ.get('APIM_AZURE_SEARCH_URL',"")
logging.basicConfig(level=LOGLEVEL)

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
# Function to generate embeddings for title and content fields, also used for query embeddings
async def generate_embeddings(text,apim_key=None):
    embeddings_config = await get_aoai_config(AZURE_OPENAI_EMBEDDING_MODEL)
    if APIM_ENABLED:
        client = AzureOpenAI(
        api_version=embeddings_config['api_version'],
        azure_endpoint=embeddings_config['endpoint'],
        api_key=apim_key
    )
    else:   
        client = AzureOpenAI(
            api_version=embeddings_config['api_version'],
            azure_endpoint=embeddings_config['endpoint'],
            azure_ad_token=embeddings_config['api_key'],
        )

    embeddings = client.embeddings.create(input=[text], model=embeddings_config['deployment']).data[0].embedding

    return embeddings

class Retrieval:
    @kernel_function(
        description="Search a knowledge base for sources to ground and give context to answer a user question. Return sources.",
        name="VectorIndexRetrieval",
    )
    async def VectorIndexRetrieval(
        self,
        input: Annotated[str, "The user question"],
        apim_key: Annotated[str, "The key to access the apim endpoint"],
        client_principal_id: Annotated[str, "The user client principal id"]
    ) -> Annotated[str, "the output is a string with the search results"]:
        search_results = []
        search_query = input
        search_filter = f"security_id/any(g:search.in(g,'{client_principal_id}'))"
        try:
            async with DefaultAzureCredential() as credential:
                start_time = time.time()
                logging.info(f"[sk_retrieval] generating question embeddings. search query: {search_query}")
                embeddings_query = await generate_embeddings(search_query,apim_key=apim_key)
                response_time = round(time.time() - start_time, 2)
                logging.info(f"[sk_retrieval] finished generating question embeddings. {response_time} seconds")
                azureSearchKey =await credential.get_token("https://search.azure.com/.default")
                azureSearchKey = azureSearchKey.token

                logging.info(f"[sk_retrieval] querying azure ai search. search query: {search_query}")
                # prepare body
                body = {
                    "select": "title, content, url, filepath, chunk_id",
                    "top": AZURE_SEARCH_TOP_K
                }
                if AZURE_SEARCH_APPROACH == TERM_SEARCH_APPROACH:
                    body["search"] = search_query
                elif AZURE_SEARCH_APPROACH == VECTOR_SEARCH_APPROACH:
                    body["vectorQueries"] = [{
                        "kind": "vector",
                        "vector": embeddings_query,
                        "fields": "contentVector",
                        "k": int(AZURE_SEARCH_TOP_K)
                    }]
                elif AZURE_SEARCH_APPROACH == HYBRID_SEARCH_APPROACH:
                    body["search"] = search_query
                    body["vectorQueries"] = [{
                        "kind": "vector",
                        "vector": embeddings_query,
                        "fields": "contentVector",
                        "k": int(AZURE_SEARCH_TOP_K)
                    }]

                if AZURE_SEARCH_USE_SEMANTIC == "true" and AZURE_SEARCH_APPROACH != VECTOR_SEARCH_APPROACH:
                    body["queryType"] = "semantic"
                    body["semanticConfiguration"] = AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG

                if AZURE_SEARCH_TRIMMING:
                    body["filter"] = search_filter

                if APIM_ENABLED:
                    headers = {
                    'Content-Type': 'application/json',
                    'api-key': apim_key
                }
                    search_endpoint = f"{APIM_AZURE_SEARCH_URL}/docs?api-version={AZURE_SEARCH_API_VERSION}"
                else:
                    headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {azureSearchKey}'
                }
                    search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    if APIM_ENABLED:
                        async with session.get(search_endpoint, headers=headers, json=body) as response:
                            status_code = response.status
                            text=await response.text()
                            json=await response.json()
                            if status_code >= 400:
                                error_on_search = True
                                error_message = f'Status code: {status_code}.'
                                if text != "": error_message += f" Error: {response.text}."
                                logging.error(f"[sk_retrieval] error {status_code} when searching documents. {error_message}")
                            else:
                                if json['value']:
                                    for doc in json['value']:
                                        search_results.append(doc['filepath'] + ": " + doc['content'].strip() + "\n")
                    else:                
                        async with session.post(search_endpoint, headers=headers, json=body) as response:
                            status_code = response.status
                            text=await response.text()
                            json=await response.json()
                            if status_code >= 400:
                                error_on_search = True
                                error_message = f'Status code: {status_code}.'
                                if text != "": error_message += f" Error: {response.text}."
                                logging.error(f"[sk_retrieval] error {status_code} when searching documents. {error_message}")
                            else:
                                if json['value']:
                                    for doc in json['value']:
                                        search_results.append(doc['filepath'] + ": " + doc['content'].strip() + "\n")

                response_time = round(time.time() - start_time, 2)
                logging.info(f"[sk_retrieval] finished querying azure ai search. {response_time} seconds")
        except Exception as e:
            error_message = str(e)
            logging.error(f"[sk_retrieval] error when getting the answer {error_message}")

        sources = ' '.join(search_results)
        return sources

    @kernel_function(
        description="Search bing for sources to ground and give context to answer a user question. Return sources.",
        name="BingRetrieval",
    )
    async def BingRetrieval(
        self,
        input: Annotated[str, "The user question"],
        bing_api_key: Annotated[str, "The key to access the bing search"],
        bing_custom_config_id: Annotated[str, "The custom config id to access the bing search"]
    ) -> Annotated[str, "the output is a string with the search results"]:
        bing_custom_config_id = await get_secret('bingCustomConfigId')
        if(APIM_ENABLED):
            endpoint=APIM_BING_CUSTOM_SEARCH_URL
        else:
            endpoint=BING_CUSTOM_SEARCH_URL
        client = CustomSearchClient(endpoint=endpoint, credentials=CognitiveServicesCredentials(bing_api_key))
        start_time = time.time()
        web_data = client.custom_instance.search(query=input, custom_config=bing_custom_config_id, count=BING_SEARCH_TOP_K)
        bing_sources = ""
        async with aiohttp.ClientSession() as session:    
            if web_data.web_pages and hasattr(web_data.web_pages, 'value'):
                tasks= [extract_text_from_html(web,session) for web in web_data.web_pages.value]
                results = await asyncio.gather(*tasks)
                for result in results:
                    bing_sources += result[:get_possitive_int_or_default(BING_SEARCH_MAX_TOKENS, 1000)]
        logging.info(f"[sk_retrieval] finished querying bing search. {time.time()-start_time} seconds")
        return bing_sources

    @kernel_function(
        description="Search a SQL or Teradata DB for sources to ground and give context to answer a user question. Return sources.",
        name="DBRetrieval",
    )
    async def DBRetrieval(self,
                    input: Annotated[str, "The user question"]
                    ) -> Annotated[str, "the output is a string with the search results"]:
        logging.info('Python HTTP trigger function processed a request.')

        try:
            # Get OpenAI configuration
            oai_config = await get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
           
            db_top_k = get_possitive_int_or_default(DB_TOP_K, TOP_K_DEFAULT)
            max_tokens = get_possitive_int_or_default(DB_MAX_TOKENS, MAX_TOKENS_DEFAULT)
            # Connect to Key Vault and get database password
            if DB_TYPE == "sql":
                db_password = await get_secret("sqlpassword")
            elif DB_TYPE == "teradata":
                db_password = await get_secret("teradatapassword")
            else:
                logging.error(f"[DBRetrieval] Invalid db_type specified")
                return ""
            azureOpenAIKey = await get_secret("azureOpenAIKey")
            #Get table data
            try:

                # Get table information from file
                with open('db_table_info.txt', 'r') as file:
                    db_table_info = file.read()

            except FileNotFoundError:
                logging.error("[DBRetrieval] db_table_info.txt not found")
                return ""
            except Exception as e:
                logging.error(f"[DBRetrieval] Unexpected error: {e}")
                return ""
            # Log configuration variables
            logging.info(f"[{DB_TYPE} Retrieval] Server: {DB_SERVER}")
            logging.info(f"[{DB_TYPE} Retrieval] Database: {DB_DATABASE}")
            logging.info(f"[{DB_TYPE} Retrieval] Username: {DB_USERNAME}")
            logging.info(f"[{DB_TYPE} Retrieval] Password: [REDACTED]")  # Do not log the password for security
            logging.info(f"[{DB_TYPE} Retrieval] Tables Info: {db_table_info}")

            # Connect to the database
            if DB_TYPE == "sql":
                driver = '{ODBC Driver 17 for SQL Server}'
                params = urllib.parse.quote_plus(f"DRIVER={driver};SERVER={DB_SERVER};DATABASE={DB_DATABASE};UID={DB_USERNAME};PWD={db_password}")
                conn_str = f'mssql+pyodbc:///?odbc_connect={params}'
            elif DB_TYPE == "teradata":
                driver = 'Teradata'
                params = urllib.parse.quote_plus(f"DRIVER={driver};DBCNAME={DB_SERVER};DATABASE={DB_DATABASE};UID={DB_USERNAME};PWD={db_password}")
                conn_str = f'teradata:///?odbc_connect={params}'
            else:
                logging.error("[DBRetrieval] Invalid db_type specified")
                return ""
            engine = create_engine(conn_str)
            logging.info(f"[{DB_TYPE} Retrieval] Connection to database is successful")
            sql_database = SQLDatabase(engine)
            # Configure Azure OpenAI
            llm = LlamaAzureOpenAI(
                engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                model=AZURE_OPENAI_CHATGPT_MODEL,
                temperature=AZURE_OPENAI_TEMPERATURE,
                azure_endpoint=oai_config['endpoint'],
                api_key=azureOpenAIKey,
                api_version=AZURE_OPENAI_APIVERSION,
            )

            embed_model = LlamaAzureOpenAIEmbedding(
                model=AZURE_OPENAI_EMBEDDING_MODEL,
                deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                api_key=azureOpenAIKey,
                azure_endpoint=oai_config['endpoint'],
                api_version=AZURE_OPENAI_EMBEDDING_APIVERSION,
            )
            Settings.llm = llm
            Settings.embed_model = embed_model

            # Get table information
            tables_info = json.loads(db_table_info.replace("'", '"'))
            table_schema_objs = []

            for table, description in tables_info.items():
                table_schema_objs.append(
                    SQLTableSchema(
                        table_name=table,
                        context_str=description
                    )
                )

            table_node_mapping = SQLTableNodeMapping(sql_database)
            obj_index = ObjectIndex.from_objects(
                table_schema_objs,
                table_node_mapping,
                VectorStoreIndex
            )
            query_engine = SQLTableRetrieverQueryEngine(
                sql_database, obj_index.as_retriever(similarity_top_k=db_top_k)
            )
            query = input[:max_tokens]
            response =await query_engine.aquery(query)
            result = response.response
            logging.info(f"[{DB_TYPE} Retrieval] SQLQuery: {response.metadata.get('sql_query')}")
            engine.dispose()
            return result
        except EnvironmentError as e:
            logging.error(f"[{DB_TYPE} Retrieval] Environment configuration error: {e}")
            return ""
        except Exception as e:
            logging.error(f"[{DB_TYPE} Retrieval]  Unexpected error: {e}")
            return ""
