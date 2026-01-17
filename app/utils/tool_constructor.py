import inspect
import ast
import logging
import app.utils.custom_functions as functions

logger = logging.getLogger(__name__)


class LLMToolConstructor:
    """Constructs tool definitions from custom functions based on user scope."""
    
    def __init__(self, provider: str, user_type: str):
        """Initialize tool constructor.
        
        Args:
            provider: LLM provider name
            user_type: Type of user to filter tools by scope
        """
        self.provider = provider
        self.user_type = user_type
    
    def get_function_list(self) -> list:
        """Get list of functions matching the user's scope.
        
        Returns:
            List of function objects for the user type
        """
        function_list = []
        for name, obj in inspect.getmembers(functions, inspect.isfunction):
            source = inspect.getsource(obj)
            try:
                tree = ast.parse(source)
                scope_value = self._extract_scope_value(tree)
                if scope_value == self.user_type:
                    function_list.append(obj)
            except Exception:
                continue
        return function_list
    
    @staticmethod
    def _extract_scope_value(tree):
        """Extract scope variable value from AST.
        
        Args:
            tree: AST parse tree
            
        Returns:
            Scope value string or None
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "scope":
                        if isinstance(node.value, ast.Str):
                            return node.value.s
                        elif isinstance(node.value, ast.Constant):
                            return node.value.value
        return None
    
    def extract_function_metadata(self, func):
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)
        
        function_description = None
        param_descriptions = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign): 
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name == "function_description":
                            function_description = ast.literal_eval(node.value)
                        elif var_name.endswith("_description"):
                            param_name = var_name.replace("_description", "")
                            param_descriptions[param_name] = ast.literal_eval(node.value)

        params = inspect.signature(func).parameters
        param_properties = {
            param: {
                "type": "string",
                "description": param_descriptions.get(param, f"{param} parameter of {func.__name__}")
            }
            for param in params
        }

        function_metadata = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": function_description or "No description available",
                "parameters": {
                    "type": "object",
                    "properties": param_properties,
                    "required": list(params.keys()),
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        return function_metadata
    
    def tool_constructor(self) -> list:
        """Construct tool definitions from function metadata.
        
        Returns:
            List of tool definition dicts compatible with LLM APIs
        """
        function_list = self.get_function_list()
        functions_metadata = [self.extract_function_metadata(func) for func in function_list]
        return functions_metadata
    
    def main(self) -> list:
        """Main entry point to get tools for the user type.
        
        Returns:
            List of tool definitions or empty list if no tools available for user type
        """
        if self.user_type == "general":
            return self.tool_constructor()
        else:
            logger.info(f"No tools available for user type: {self.user_type}")
            return []
        