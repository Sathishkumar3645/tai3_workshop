import json
import os
import inspect
from dotenv import load_dotenv
from app.core.config import settings
import app.utils.custom_functions as functions
from app.utils.tool_execution import ExecuteTool

import httpx
from groq import Groq
from openai import OpenAI

load_dotenv()

class LLMTrigger:
    def __init__(self, provider, tools, userQuery, userType, conversationHistory, prompt):
        self.provider = provider
        self.tools = tools
        self.tool_call_identified = True
        self.userQuery = userQuery
        self.userType = userType
        self.messages = None
        self.conversationHistory = conversationHistory
        # Use httpx client with verify=False for Groq (corporate proxy/SSL inspection)
        http_client = httpx.Client(verify=False)
        self.groqClient = Groq(api_key=os.getenv("groq_api_key"), http_client=http_client)
        self.openaiClient = OpenAI(api_key=os.getenv("openai_api_key"))
        self.prompt = prompt
        self.configData = settings  # Use settings from app.core.config
    
    def functionCollector(self, module):
        import ast
        func_dict = {}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            source = inspect.getsource(obj)
            try:
                tree = ast.parse(source)
                scope_value = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "scope":
                                if isinstance(node.value, ast.Str):
                                    scope_value = node.value.s
                                elif isinstance(node.value, ast.Constant):
                                    scope_value = node.value.value
                if scope_value == self.userType:
                    func_dict[name] = obj
            except Exception:
                continue
        return func_dict
    
    def messageConstructor(self, prompt):
        self.messages = [
                    {"role": "system", "content": "You are an supportive e commerce assitant, who helps customers find products and answer questions related to the products. Use the information provided in the conversation history and product catalog to assist the user effectively."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
        return self.messages
    
    def openaicall(self):
        """Call OpenAI API with tool support and error handling."""
        final_response = None
        messages = self.messageConstructor(self.prompt)
        retry_count = 0
        max_retries = 2
        
        try:
            while self.tool_call_identified:
                try:
                    response = self.openaiClient.chat.completions.create(
                        model=self.configData.OPENAI_MODEL_NAME,
                        messages=messages,
                        tools=self.tools if self.tools else None,
                        tool_choice="auto" if self.tools else None,
                        # max_tokens=getattr(self.configData, 'MAX_TOKENS', 1024)
                    )
                except Exception as api_error:
                    if "tool" in str(api_error).lower() and retry_count < max_retries:
                        retry_count += 1
                        messages.append({
                            "role": "assistant",
                            "content": "I attempted to call a function but encountered a formatting issue. Let me try a different approach without using the tool."
                        })
                        continue
                    raise api_error
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                
                if tool_calls:
                    available_functions = self.functionCollector(functions)
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                    "type": tool_call.type,
                                }
                                for tool_call in tool_calls
                            ],
                        }
                    )
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = ExecuteTool(function_name, function_args, available_functions).mainExecution()
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                else:
                    self.tool_call_identified = False
                    final_response = response.choices[0].message.content
        except Exception as e:
            final_response = f"Error: {str(e)}"
        return final_response
    
    def groqCall(self):
        """Call Groq API with tool support and error handling."""
        final_response = None
        messages = self.messageConstructor(self.prompt)
        retry_count = 0
        max_retries = 2
        
        try:
            while self.tool_call_identified:
                try:
                    response = self.groqClient.chat.completions.create(
                        model=self.configData.GROQ_MODEL_NAME,
                        messages=messages,
                        tools=self.tools,
                        tool_choice=getattr(self.configData, 'TOOL_CHOICE', None),
                        max_tokens=getattr(self.configData, 'MAX_TOKENS', 1024),
                        temperature=getattr(self.configData, 'TEMPERATURE', 0.7)
                    )
                except Exception as api_error:
                    if "Failed to call a function" in str(api_error) and retry_count < max_retries:
                        retry_count += 1
                        messages.append({
                            "role": "assistant",
                            "content": "I attempted to call a function but encountered a formatting issue. Let me try a different approach without using the tool."
                        })
                        continue
                    raise api_error
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                if tool_calls:
                    available_functions = self.functionCollector(functions)
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                    "type": tool_call.type,
                                }
                                for tool_call in tool_calls
                            ],
                        }
                    )
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = ExecuteTool(function_name, function_args, available_functions).mainExecution()
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                else:
                    if '</function>' in response.choices[0].message.content:
                        pass
                    else:
                        self.tool_call_identified = False
                        final_response = response.choices[0].message.content
        except Exception as e:
            final_response = f"Error: {str(e)}"
        return final_response
     
    def main(self):
        if self.provider == "groq":
            return self.groqCall()
        elif self.provider == "openai":
            return self.openaicall()