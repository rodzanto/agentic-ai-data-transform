"""
ETL Agent using Amazon Bedrock (Claude) for SparkSQL transformations

This script implements a chatbot that:
1. Takes natural language ETL transformation requests
2. Uses Claude to generate SparkSQL queries
3. Processes uploaded files directly
4. Returns the results to the user

Requirements:
- boto3
- streamlit (for web interface)
- pandas (for data handling)
"""

import boto3
import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
import re
import asyncio
import uuid
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Amazon Bedrock configuration
BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0" 

class ETLAgent:
    """ETL Agent that uses Claude to generate SparkSQL and process uploaded files."""
    
    def __init__(self):
        """Initialize the ETL Agent with Bedrock client and config."""
        self.bedrock_client = boto3.client('bedrock-runtime')
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as f:
            self.config = json.load(f)
        self.s3_bucket = self.config['s3_bucket']
        self.temp_table = None
        self.s3_path = None
        self.mcp_server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@lishenxydlgzs/aws-athena-mcp"],
            env={
                "OUTPUT_S3_PATH": os.environ.get("OUTPUT_S3_PATH", "s3://rodzanto2024-uswest2/athena/"),
                "AWS_REGION": os.environ.get("AWS_REGION", "us-west-2"),
            }
        )

    async def _run_query_async(self, query, database="data_transactions_db", maxRows=1000, timeoutMs=60000):
        logger.info(f"[MCP] Sending query to Athena MCP Server: {query}\nDatabase: {database}, maxRows: {maxRows}, timeoutMs: {timeoutMs}")
        async with stdio_client(self.mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool('run_query', {
                    'database': database,
                    'query': query,
                    'maxRows': maxRows,
                    'timeoutMs': timeoutMs
                })
                logger.info(f"[MCP] Received response from Athena MCP Server: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
                return result

    def mcp_run_query(self, query, database="data_transactions_db", maxRows=1000, timeoutMs=60000):
        logger.info(f"[MCP] mcp_run_query called with query: {query}\nDatabase: {database}, maxRows: {maxRows}, timeoutMs: {timeoutMs}")
        result = asyncio.run(self._run_query_async(query, database, maxRows, timeoutMs))
        logger.info(f"[MCP] mcp_run_query result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
        return result

    def upload_file_to_s3(self, file_info):
        """Upload file to S3 and return S3 URI."""
        s3 = boto3.client('s3')
        file_name = file_info['name']
        bucket = self.s3_bucket.replace('s3://', '').split('/')[0]
        prefix = '/'.join(self.s3_bucket.replace('s3://', '').split('/')[1:])
        key = f"{prefix.rstrip('/')}/{file_name}" if prefix else file_name
        s3.put_object(Bucket=bucket, Key=key, Body=file_info['content'])
        s3_uri = f"s3://{bucket}/{key}"
        self.s3_path = s3_uri
        return s3_uri

    def to_dict(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        else:
            return dict(obj)

    def infer_csv_schema(self, file_content):
        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(file_content), nrows=0)
        columns = df.columns.tolist()
        # All columns as string (lowercase) for Athena, use unquoted identifiers
        schema = ', '.join([f'{col} string' for col in columns])
        return schema

    def get_s3_dir(self, s3_uri):
        # Remove the file name from the S3 URI
        parsed = urlparse(s3_uri)
        path = parsed.path
        if path.endswith('/'):
            return s3_uri
        else:
            dir_path = '/'.join(path.split('/')[:-1])
            return f's3://{parsed.netloc}{dir_path}/'

    def normalize_athena_result(self, result):
        """Normalize Athena MCP result to always have 'rows' and 'columns' keys for Streamlit display."""
        # If result is already in the right format
        if isinstance(result, dict):
            if 'rows' in result and 'columns' in result:
                return result
            # If result is a dict with 'content' key (MCP format)
            if 'content' in result and isinstance(result['content'], list):
                for block in result['content']:
                    # Handle TextContent with JSON in 'text'
                    if hasattr(block, 'text'):
                        try:
                            parsed = json.loads(block.text)
                            if 'rows' in parsed and 'columns' in parsed:
                                return {
                                    'rows': parsed['rows'],
                                    'columns': parsed['columns']
                                }
                        except Exception:
                            continue
                    # Handle dicts with 'text' key
                    if isinstance(block, dict) and 'text' in block:
                        try:
                            parsed = json.loads(block['text'])
                            if 'rows' in parsed and 'columns' in parsed:
                                return {
                                    'rows': parsed['rows'],
                                    'columns': parsed['columns']
                                }
                        except Exception:
                            continue
                    # Handle table type (legacy)
                    if isinstance(block, dict) and block.get('type') == 'table':
                        return {
                            'rows': block.get('rows', []),
                            'columns': block.get('columns', [])
                        }
        # Fallback: return empty
        return {'rows': [], 'columns': []}

    def process_user_request(self, transformation_prompt: str, files: list = None, confirm_cleanup: bool = False):
        """Process user request: upload file, create table, preview, transform, cleanup."""
        try:
            if confirm_cleanup and self.temp_table:
                # User confirmed, drop temp table
                drop_query = f"DROP TABLE IF EXISTS {self.temp_table}"
                drop_result = self.mcp_run_query(drop_query)
                self.temp_table = None
                return {'status': 'cleanup', 'message': 'Temporary table deleted.', 'drop_result': self.to_dict(drop_result)}

            if not files or len(files) == 0:
                return {'status': 'error', 'message': 'No file uploaded.'}
            file_info = files[0]  # Only support one file for now
            # 1. Upload file to S3
            s3_uri = self.upload_file_to_s3(file_info)
            s3_dir = self.get_s3_dir(s3_uri)
            # 2. Create temp table in Athena (infer table name from file)
            table_name = f"temp_{uuid.uuid4().hex[:8]}"
            db_table_name = f"data_transactions_db.{table_name}"
            self.temp_table = table_name
            file_format = 'CSV' if file_info['type'] in ['text/csv', 'application/vnd.ms-excel'] else 'JSON'
            # Athena CREATE TABLE statement using correct Athena SQL syntax for CSV
            if file_format == 'CSV':
                schema = self.infer_csv_schema(file_info['content'])
                create_table_query = f'''
CREATE EXTERNAL TABLE {db_table_name} (
  {schema}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"'
)
STORED AS TEXTFILE
LOCATION '{s3_dir}'
TBLPROPERTIES ('skip.header.line.count'='1');
'''
            else:
                # For JSON, fallback to a single column
                create_table_query = f'''
CREATE TABLE {db_table_name} (
  data string
)
WITH (
  external_location = '{s3_dir}',
  format = 'JSON'
)
'''
            logger.info(f"Athena CREATE TABLE query for {db_table_name}:")
            logger.info(create_table_query)
            try:
                create_result = self.mcp_run_query(create_table_query)
                logger.info(f"Athena CREATE TABLE result for {db_table_name}: {create_result}")
                if getattr(create_result, 'isError', False) or (isinstance(create_result, dict) and create_result.get('isError')):
                    raise Exception(str(create_result))
            except Exception as e:
                logger.error(f"CREATE TABLE failed: {e}")
                return {'status': 'error', 'message': f'Failed to create table: {e}'}
            # 3. Preview table
            preview_query = f"SELECT * FROM {db_table_name} LIMIT 5"
            preview_result_raw = self.to_dict(self.mcp_run_query(preview_query))
            preview_result = self.normalize_athena_result(preview_result_raw)
            # 4. Get schema/sample for Bedrock prompt
            schema_query = f"SHOW COLUMNS IN {db_table_name}"
            schema_result = self.to_dict(self.mcp_run_query(schema_query))
            # 5. Call Bedrock to generate transformation query
            bedrock_prompt = f"Given the table '{db_table_name}' with schema: {schema_result} and preview: {preview_result}, {transformation_prompt}"
            bedrock_response = self._call_claude(bedrock_prompt)
            logger.info(f"Assistant/Claude response: {json.dumps(bedrock_response, indent=2) if isinstance(bedrock_response, dict) else bedrock_response}")
            # 6. Extract SparkSQL query from Bedrock response
            output_message = bedrock_response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            spark_sql = None
            for block in content_blocks:
                if '```sql' in block.get('text', ''):
                    # Extract SQL from markdown
                    match = re.search(r'```sql(.*?)```', block['text'], re.DOTALL)
                    if match:
                        spark_sql = match.group(1).strip()
                        break
            if not spark_sql:
                # Fallback: use first code block or text
                spark_sql = content_blocks[0].get('text', '') if content_blocks else None
            if not spark_sql:
                return {'status': 'error', 'message': 'Could not extract SparkSQL from Bedrock response.'}

            # --- BEGIN: Robustify transformation SQL for quantity column ---
            # If the transformation query casts 'quantity' to DOUBLE or DECIMAL, make it robust to empty strings
            # Patterns to match: CAST(quantity AS DOUBLE), CAST(quantity AS DECIMAL(10,2)), etc.
            cast_patterns = [
                (r'CAST\(quantity AS DOUBLE\)', "CAST(NULLIF(quantity, '') AS DOUBLE)"),
                (r'CAST\(quantity AS DECIMAL\(10,2\)\)', "CAST(NULLIF(quantity, '') AS DECIMAL(10,2))")
            ]
            for pattern, replacement in cast_patterns:
                if re.search(pattern, spark_sql):
                    spark_sql = re.sub(pattern, replacement, spark_sql)
                    # Try to add a WHERE clause to filter out empty or null quantity values
                    if 'WHERE' not in spark_sql:
                        # Find FROM ... and insert WHERE before the semicolon (if present)
                        from_pattern = r'(FROM\s+\S+)(\s*)(;|$)'
                        match = re.search(from_pattern, spark_sql, re.IGNORECASE)
                        if match:
                            from_clause = match.group(1)
                            whitespace = match.group(2)
                            semicolon_or_end = match.group(3)
                            # Insert WHERE clause before the semicolon or at the end
                            spark_sql = re.sub(
                                from_pattern,
                                f"{from_clause} WHERE quantity IS NOT NULL AND quantity <> ''{whitespace}{semicolon_or_end}",
                                spark_sql,
                                flags=re.IGNORECASE
                            )
            # --- END: Robustify transformation SQL for quantity column ---

            # 7. Run transformation query
            result_raw = self.to_dict(self.mcp_run_query(spark_sql))
            # Error handling: check for Athena errors in the result
            if (
                (isinstance(result_raw, dict) and result_raw.get('isError')) or
                (isinstance(result_raw, dict) and 'content' in result_raw and isinstance(result_raw['content'], list) and any('Error:' in (block.get('text', '') if isinstance(block, dict) else str(block)) for block in result_raw['content']))
            ):
                # Try to extract error message
                error_message = None
                if isinstance(result_raw, dict) and 'content' in result_raw:
                    for block in result_raw['content']:
                        if isinstance(block, dict) and 'text' in block and 'Error:' in block['text']:
                            error_message = block['text']
                            break
                        elif isinstance(block, str) and 'Error:' in block:
                            error_message = block
                            break
                if not error_message:
                    error_message = str(result_raw)
                return {'status': 'error', 'message': f'Athena error: {error_message}'}
            result = self.normalize_athena_result(result_raw)
            logger.info(f"Transformation result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
            return {
                'status': 'success',
                'preview': preview_result,
                'schema': schema_result,
                'transformation_query': spark_sql,
                'result': result,
                'table_name': db_table_name
            }
        except Exception as e:
            logger.error(f"Error in process_user_request: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def execute_spark_sql(self, query: str, description: str) -> Dict[str, Any]:
        """Execute a SparkSQL query on the uploaded data."""
        try:
            logger.info(f"Executing SparkSQL query: {description}")
            
            # For now, return a dummy response
            dummy_response = {
                "status": "success",
                "message": "This is a simulated response for the uploaded data",
                "query": query,
                "description": description,
                "result": {
                    "columns": ["id", "name", "value"],
                    "data": [
                        [1, "Item 1", 100],
                        [2, "Item 2", 200],
                        [3, "Item 3", 300]
                    ]
                }
            }
            return dummy_response
        except Exception as e:
            logger.error(f"Error executing SparkSQL query: {e}")
            return {"error": str(e)}
    
    def _call_claude(self, user_message: str, files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call Claude through Amazon Bedrock Converse API with the user message and file attachments."""
        try:
            # Prepare the message content
            content = []
            # Build the input data section
            input_data_content = []
            if files:
                for file_info in files:
                    input_data_content.append(f"""<file name=\"{file_info['name']}\" type=\"{file_info['type']}\">\n<metadata>\n    name: {file_info['name']}\n    type: {file_info['type']}\n</metadata>\n<raw_data>\n{file_info['content']}\n</raw_data>\n</file>""")
            prompt = f"""Consider the input data in the <input_data> XML tags below for processing:\n<input_data>\n{chr(10).join(input_data_content)}\n</input_data>\n\n{user_message}"""
            content.append({"text": prompt})
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            system = [
                {
                    "text": """ETL assistant using Amazon Athena SQL. Your task is to:\n1. Analyze the input data provided in the <input_data> tags\n2. Generate an appropriate Athena SQL query based on the data structure and user request\n3. Return ONLY a single, complete Athena SQL statement (no comments, no extra statements, no multi-statement blocks, no markdown, no explanations)\n4. Do NOT include any comments, explanations, or additional queries.\n\nWrite efficient, optimized Athena SQL queries. Handle edge cases and validate inputs. Do NOT use SparkSQL syntax. Use only valid Amazon Athena SQL syntax.\n\nIMPORTANT: Only output a single Athena SQL statement that can be executed as-is. Do NOT include comments, markdown, or more than one statement.\n\nWhen normalizing or casting the quantity column to a numeric type, ALWAYS filter out rows where quantity is null or an empty string (e.g., WHERE TRIM(quantity) <> '' AND quantity IS NOT NULL) before casting. This should be part of the SQL statement you return."""
                }
            ]
            inference_config = {
                "maxTokens": 8000,
                "temperature": 0
            }
            request_payload = {
                "modelId": BEDROCK_MODEL_ID,
                "messages": messages,
                "system": system,
                "inferenceConfig": inference_config
            }
            logger.info("Request payload sizes:")
            logger.info(f"User message length: {len(user_message)} characters")
            if files:
                for file_info in files:
                    logger.info(f"File '{file_info['name']}' content length: {len(file_info['content'])} characters")
            logger.info("Request payload (excluding file content):")
            logger.info(json.dumps({
                "modelId": BEDROCK_MODEL_ID,
                "messages": [{
                    "role": msg["role"],
                    "content": [
                        {"text": c["text"]}
                        for c in msg["content"]
                    ]
                } for msg in messages],
                "system": system,
                "inferenceConfig": inference_config
            }, indent=2))
            response = self.bedrock_client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=messages,
                system=system,
                inferenceConfig=inference_config
            )
            logger.info(f"Raw Claude/Bedrock response: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")
            return response
        except Exception as e:
            logger.error(f"Error calling Claude with Converse API: {e}")
            raise
    
    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tool calls generated by Claude via Converse API."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("toolUseId")
            tool_spec = tool_call.get("toolSpec", {}).get("name")
            tool_input = tool_call.get("input", {})
            
            result = {"toolUseId": tool_name}
            
            try:
                if tool_spec == "execute_spark_sql":
                    query = tool_input.get("query")
                    description = tool_input.get("description")
                    output = self.execute_spark_sql(query, description)
                    result["output"] = output
                else:
                    result["output"] = {"error": f"Unknown tool: {tool_spec}"}
            
            except Exception as e:
                logger.error(f"Error processing tool call {tool_spec}: {e}")
                result["output"] = {"error": str(e)}
            
            tool_results.append(result)
        
        return tool_results
    
    def _get_final_response(self, user_message: str, initial_response: Dict[str, Any], 
                           tool_results: List[Dict[str, Any]]) -> str:
        """Get the final response from Claude with the tool results using Converse API."""
        try:
            # Get the conversation history from the initial response
            initial_messages = [
                {
                    "role": "user",
                    "content": [{"text": user_message}]
                },
                initial_response.get("output", {}).get("message", {})
            ]
            
            # Create a system prompt as a list of text blocks - more concise version
            system = [
                {
                    "text": """ETL assistant. Explain transformation results clearly. Handle errors gracefully."""
                }
            ]
            
            # Configure inference parameters
            inference_config = {
                "maxTokens": 8000,
                "temperature": 0
            }
            
            # Call Claude through Bedrock Converse API with the tool results
            response = self.bedrock_client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=initial_messages,
                system=system,
                inferenceConfig=inference_config,
                toolResults=tool_results
            )
            
            # Extract and return the text response
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            # Extract text from content blocks
            if content_blocks:
                return "\n".join([block.get("text", "") for block in content_blocks if "text" in block])
            return "No response content available."
        except Exception as e:
            logger.error(f"Error getting final response: {e}")
            raise

def main():
    """Main function to run the ETL agent."""
    etl_agent = ETLAgent()
    
    print("ETL Agent initialized. Type 'exit' to quit.")
    print("Example: 'Transform the uploaded CSV file to normalize the values'")
    
    while True:
        user_input = input("\nEnter your request: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = etl_agent.process_user_request(user_input)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()
