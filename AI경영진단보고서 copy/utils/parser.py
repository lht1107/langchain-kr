from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# 응답 스키마 정의
response_schemas = [
    ResponseSchema(name="detailed_analysis",
                   description="The detailed analysis in English"),
    ResponseSchema(name="final_report",
                   description="The final report in Korean")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
