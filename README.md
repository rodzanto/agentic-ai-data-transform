# Agentic AI Data Transform

A reference architecture and code samples for building intelligent data transformation pipelines using natural language instructions with Amazon Bedrock Agents, Amazon EMR, and MCP (Model-Container-Protocol).

![Architecture Diagram](docs/images/architecture-diagram.png)

## Overview

This repository demonstrates how to build an ETL (Extract, Transform, Load) system that allows users to transform structured data using natural language instructions. For example, a user can input commands like "Normalize all numerical values in this dataset" and the system will automatically generate and execute the appropriate SparkSQL queries.

The solution leverages:
- **Amazon Bedrock Agents** with Claude 3.7 for natural language understanding and orchestration
- **Model-Context-Protocol (MCP)** server to interface between the AI agent and data processing engines
- **Amazon EMR/Glue** with SparkSQL for scalable data processing
- **Amazon S3** for data storage

## Key Features

- 🗣️ **Natural Language Interface**: Transform your data using everyday language
- 🧠 **Intelligent Orchestration**: Claude 3.7 plans and executes complex data transformations
- ⚡ **Scalable Processing**: Leverage Apache Spark for processing large datasets
- 🔄 **Dynamic Adaptation**: System adjusts to different datasets and transformation needs
- 🔌 **Modular Architecture**: Easily swap components or extend functionality

## Getting Started

### Prerequisites

- AWS Account with access to Bedrock, EMR, and S3
- Python 3.8+
- AWS CLI configured

### Installation

```bash
# Clone the repository
git clone https://github.com/aws-samples/agentic-ai-data-transform.git
cd agentic-ai-data-transform

# TODO: Add installation steps for deploying the required AWS resources
# This will include CloudFormation/CDK deployment for Bedrock Agents,
# MCP Server, and EMR/Glue components
```

### Configuration

The solution is primarily configured through Amazon Bedrock Agents:

1. TODO: Add configuration steps for Bedrock Agents
2. TODO: Add configuration for MCP Server and SparkSQL integration
3. TODO: Add data source configuration guidance

## Usage Examples

### Basic Data Normalization

```
# Example request to the Bedrock Agent via API
POST /agents/{agentId}/text
{
  "inputText": "Normalize all numerical columns in the customer_data dataset",
  "parameters": {
    "sourceDataset": "s3://my-bucket/data/customer_data.csv",
    "destination": "s3://my-bucket/data/normalized_customer_data"
  }
}
```

### Complex Transformations

```
# Example request for multiple transformation operations
POST /agents/{agentId}/text
{
  "inputText": "For the sales_transactions dataset: 
                1. Remove all duplicates
                2. Convert dates to ISO format
                3. Calculate a new column with 7-day moving average for sales
                4. Filter out transactions below $10",
  "parameters": {
    "sourceDataset": "s3://my-bucket/data/sales_transactions.parquet",
    "destination": "s3://my-bucket/data/processed_sales"
  }
}
```

## Architecture Details

### Components

1. **User Interface**: Accepts natural language instructions from users
2. **API Gateway**: Routes requests to Bedrock Agents
3. **Bedrock Agents**: 
   - Parses natural language instructions
   - Plans transformation steps
   - Generates SparkSQL queries
   - Orchestrates the entire process
4. **MCP Server**: 
   - Provides standardized interface between AI and data systems
   - Routes queries to the appropriate processing engine
5. **EMR/Glue with SparkSQL**: Executes data transformations at scale
6. **Data Catalog**: Maintains metadata about available datasets
7. **S3 Storage**: Stores raw and processed data

### Workflow

1. User submits a natural language instruction
2. Bedrock Agent interprets the request
3. Agent generates appropriate SparkSQL queries
4. Queries are sent to the MCP Server
5. MCP Server routes queries to SparkSQL
6. Results are stored in the destination location
7. Agent provides a human-readable summary back to the user

## Customization

### Adding New Transformation Types

1. Add new examples to the `examples/transformations/` directory
2. Update the prompt templates in `prompts/transformation_templates.json`
3. If needed, add new SparkSQL templates in `config/sql_templates.json`

### Supporting New Data Sources

1. Add connector configuration in `config/data_sources.json`
2. Implement the connector interface in `src/connectors/`
3. Update the agent configuration to recognize the new data source

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Security

See [SECURITY.md](SECURITY.md) for details on our security policy and how to report security issues.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
