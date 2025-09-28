import azure.functions as func
import os
from ast import literal_eval
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from json import loads, dumps
from llama_index.core import Settings, PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorContextRetriever
# from custom_graph_retrievers import *
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from logging import basicConfig, INFO
from pyodbc import connect
from sql_metadata import Parser

load_dotenv()

NLTK_DATA = os.getenv("NLTK_DATA")
TIKTOKEN_CACHE = os.getenv("TIKTOKEN_CACHE")

from llama_index.llms.azure_openai import AzureOpenAI

AZURE_BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
SEMANTIC_CONTAINER_NAME = os.getenv("SEMANTIC_CONTAINER_NAME")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ENGINE = os.getenv("AZURE_OPENAI_ENGINE")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_TYPE = os.getenv("AZURE_OPENAI_TYPE")
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_ENGINE = os.getenv("AZURE_OPENAI_EMBEDDING_ENGINE")

SERVER = os.getenv("SERVER")
DATABASE = os.getenv("DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("PASSWORD")
DRIVER = os.getenv("DRIVER")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")

COLUMN_ALIASES = {"LocationName": "LatestLocation"}
store_name = "SILVER LAKE MALL"

example_query = "SELECT CompanyDivisionCode, COUNT(StoreNumber) AS StoreCount FROM Location WHERE Country = 'USA' GROUP BY CompanyDivisionCode;"
example_query_modified = "SELECT dim.Location.CompanyDivisionCode, COUNT(dim.Location.StoreNumber) AS StoreCount FROM dim.Location WHERE dim.Location.Country = 'USA' GROUP BY dim.Location.CompanyDivisionCode;"

basicConfig(
    level = INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

llm = AzureOpenAI(
    model = AZURE_OPENAI_MODEL_NAME,
    engine = AZURE_OPENAI_ENGINE,
    api_key = AZURE_OPENAI_KEY,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_type = AZURE_OPENAI_TYPE,
    api_version = "2024-03-01-preview",
    temperature = 0.3,
)

embeddings = AzureOpenAIEmbedding(
    model = AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    deployment_name = AZURE_OPENAI_EMBEDDING_ENGINE,
    api_key = AZURE_OPENAI_KEY,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_version = "2024-02-01",
)

Settings.llm = llm
Settings.embed_model = embeddings


# @app.route(route="http_trigger", auth_level=func.AuthLevel.FUNCTION)
def main(req: func.HttpRequest) -> func.HttpResponse:

    req_json = req.get_json()
    db_name = req_json["database"]
    query = req_json["query"]

    try:
        blob_name = f"{db_name}_semantic.json"
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
        container_client = blob_service_client.get_container_client(SEMANTIC_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        semantic = loads(blob_data)
    except Exception as e:
         return func.HttpResponse(f"Error in loading from Blob Storage: {str(e)}", status_code = 500)

    failsafe_dict = {}
    for schema in semantic.keys():
        failsafe_dict[schema] = []
        for table_data in semantic[schema]["tables"]:
            table = next(iter(table_data))
            failsafe_dict[schema].append(table)

    neo4jpg = Neo4jPropertyGraphStore(
        username = NEO4J_USERNAME,
        password = NEO4J_PASSWORD,
        url = NEO4J_URI
    )

    index = PropertyGraphIndex.from_existing(neo4jpg)

    retriever1 = VectorContextRetriever(
        graph_store = neo4jpg,
        include_text = True,
        # include_properties = True
    )

    query_engine = RetrieverQueryEngine.from_args(
        index.as_retriever(
            include_text = True,
            sub_retrievers = [retriever1]
        )
    )

    tables = query_engine.query(f"""
        determine the nodes with "type" property TABLE from the knowledge graph that 
        can be queried to answer the following business question: {query}\n\n
        explore the "definition" property of the nodes to select the relevant tables\n
        the output must be a list, with each element being the table chosen to 
        answer the business question\n
        do not add any extra information to the output or to the names of the selected tables\n
        do not create any fictitious table names\n
        for example, if the tables Sale and Currency are selected, then the output 
        must look like this: ['Sale','Currency']
    """)
    tables = literal_eval(tables.response)

    if "silver lake mall" in query.lower():
        pseudo_join = query_engine.query(f"""
            determine the nodes with "type" property TABLE from the knowledge graph that can 
            be used to connect the latest information about 'SILVER LAKE MALL' with {tables}\n\n
            explore the "definition" property of the nodes to select the relevant tables\n
            the output must be a list, with each element being the table chosen to 
            answer the business question\n
            do not add any extra information to the output or to the names of the selected tables\n
            do not create any fictitious table names\n
            for example, if the tables chosen to make the connection were LocationLatest and CalendarKey 
            then the output must look like this: ['LocationLatest','CalendarKey']
        """)
        pseudo_join = literal_eval(pseudo_join.response)

        sql_query = query_engine.query(f"""
            the following is a business question that must be answered 
            using an SQL query: {query}\n\n
            the following list provides the tables that contain columns that can be relevant 
            in answering the business question: {tables}\n\n
            the following tables have been chosen to connect the latest information about 
            'SILVER LAKE MALL' with the aforementioned tables: {pseudo_join}\n\n
            explore each table's HAS_COLUMN relationships to determine the nodes with "type" 
            property COLUMN that can be queried to answer the business question\n
            use the "definition" property of each column to select the relevant columns\n
            generate a syntactically correct SQL query to answer the business question\n
            do not use NATURAL JOIN, use JOIN-ON statements if a JOIN operation is necessary\n
            do not mention any fictitious tables and columns that are not present in the knowledge graph\n
            respond with the generated SQL query, do not give any alternate queries or any other information
        """)

    else:
        sql_query = query_engine.query(f"""
            the following is a business question that must be answered 
            using an SQL query: {query}\n\n
            the following list provides the tables that contain columns that can be relevant 
            in answering the business question: {tables}\n\n
            explore each table's HAS_COLUMN relationships to determine the nodes with "type" 
            property COLUMN that can be queried to answer the business question\n
            use the "definition" property of each column to select the relevant columns\n
            generate a syntactically correct SQL query to answer the business question\n
            do not use NATURAL JOIN, use JOIN-ON statements if a JOIN operation is necessary\n
            do not mention any fictitious tables and columns that are not present in the knowledge graph\n
            respond with the generated SQL query, do not give any alternate queries or any other information
        """)

    corrected_query = str(sql_query)

    if corrected_query.startswith("```sql"):
        corrected_query = corrected_query[7:-3]

    graph_query_parser = Parser(sql=corrected_query)
    graph_tables_list = list(graph_query_parser.tables)
    graph_columns_list = list(graph_query_parser.columns)
    graph_columns_list = [val.split(".")[-1] for val in graph_columns_list]
    graph_column_dict = graph_query_parser.columns_dict
    graph_column_dict = {u: [val.split(".")[-1] for val in v] for (u, v) in graph_column_dict.items()}

    for k, v in COLUMN_ALIASES.items():
        if k.lower() in [col.lower() for col in graph_columns_list]:
            corrected_query = corrected_query.replace(k, v)

    chosen_tables = []
    for table in graph_tables_list:
        for schema, schema_tables in failsafe_dict.items():
            if table in schema_tables:
                chosen_tables.append(str(schema + "." + table))

    modified_query = llm.complete(f"""
        the following is a business question that needs to be answered using an SQL 
        query executed against Claire's Accessories' enterprise database: {query}\n\n
        the following SQL query was generated to answer the business question: {corrected_query}\n\n
        the following tables were queried to answer the business question: {graph_tables_list}\n\n
        modify the SQL query by replacing the table names with their syntactically correct names: {chosen_tables}\n\n
        do not use any aliases for the table names\n
        the column names must also be adjusted to accommodate the syntactically correct table names\n
        for example, if the SQL query was: {example_query}\n\n
        then the modified SQL query should look like: {example_query_modified}\n\n
        do not mention any fictitious tables and columns that were not present in the original SQL query\n
        respond with the modified SQL query only, do not give any alternate queries or any other information
    """).text

    if "where" in graph_column_dict.keys():
        if 'CalendarKey' in graph_column_dict["where"] or 'CalendarKey' in graph_column_dict["select"]:
            where_clause_value = '(SELECT CalendarKey FROM dim.Calendar WHERE CalendarDate=$value$)'
            modified_query = llm.complete(f"""
                the following SQL query has been generated to answer a business question: {modified_query}\n\n
                in this SQL query, replace the values in the where clause involving the CalendarKey 
                column with the following statement: {where_clause_value}\n\n
                replace $value$ with the values on the right hand side of the WHERE condition being modified\n
                make similar changes if a SELECT CASE statement is present in the query\n
                do not make any other modifications to the sql query\n
                respond with the modified SQL query, the output must not contain any alternate queries 
                or any other information\n
                do not nest the rest of the statements from the SQL queries with the WHERE clause
            """).text

        if 'LocationLatestKey' in graph_column_dict["where"] or 'LocationLatestKey' in graph_column_dict["select"]:
            where_clause_value = f"(SELECT LocationLatestKey FROM dim.LocationLatest WHERE Location='{store_name}')"
            modified_query = llm.complete(f"""
                the following SQL query has been generated to answer a business question: {modified_query}\n\n
                in this SQL query, replace the values in the where clause involving the LocationLatestKey 
                column with the following statement: {where_clause_value}\n\n
                make similar changes if a SELECT CASE statement is present in the query\n
                do not make any other modifications to the sql query\n
                respond with the modified SQL query, the output must not contain any alternate queries 
                or any other information\n
                do not nest the rest of the statements from the SQL queries with the WHERE clause
            """).text

    if "SalesTransactions" in modified_query:
        modified_query = modified_query.replace("SalesTransactions", "Sale")

    if "Silver Lake Mall" in modified_query:
        modified_query = modified_query.replace("Silver Lake Mall", "SILVER LAKE MALL")

    if modified_query.startswith("```sql"):
        modified_query = modified_query[7:-3]

    sqlserver_query = llm.complete(f"""
        the following SQL query has been generated for a MySQL database: {modified_query}\n\n
        convert it into an SQL Server query so that it can be used by PyODBC\n
        do not change the names of the tables or columns\n
        only make syntactical changes to the SQL query to tailor it for Microsoft ODBC 18\n
        do not provide any extra information or description, or add any unnecessary punctuation or quotation marks\n
        the new SQL Server query must be the only output\n
    """).text

    result_dict = {'tables': graph_tables_list, 'columns': graph_columns_list}

    try:
        conn = connect(
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={SERVER},1433;"
            f"Database={DATABASE};"
            f"UID={DB_USERNAME};"
            f"PWD={PASSWORD};"
            "TrustServerCertificate=yes;"
        )
        cursor = conn.cursor()
        cursor.execute(sqlserver_query)
        output = cursor.fetchall()
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        return func.HttpResponse(f"The following error was encountered: {e}", status_code=500)

    natural_language_response = llm.complete(f"""
            the following business question was answered using an SQL Server query: {query}\n\n
            this was the data table that was returned to answer the question: {output}\n\n
            generate a natural language response to the business question based on its output data table\n\n
            do not perform any additional arithmetic operations on the data\n
            do not provide any follow-up comments as part of the natural language response\n
            the response must be summarised in less than 80 words\n
            do not explain the columns belonging to the output data table, elaborate on the data returned
    """).text
    natural_language_response = natural_language_response.replace("$", "USD ")


    return func.HttpResponse(dumps(
        {
            "Engine Response": natural_language_response,
            "Identified Attributes": result_dict,
            "Query SQL": sqlserver_query,
        }
    ), status_code=200)
