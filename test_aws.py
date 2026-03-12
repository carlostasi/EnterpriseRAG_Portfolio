import boto3
from langchain_aws import ChatBedrock

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

llm = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", 
    model_kwargs={"temperature": 0.5}
)

print("Invio una richiesta di prova ad AWS Bedrock...")
response = llm.invoke("Ciao! Sei connesso tramite AWS Bedrock. Dammi cenno di vita")

print("\nRisposta del Cloud:")
print(response.content)
