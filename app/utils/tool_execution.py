class ExecuteTool:
    def __init__(self, functionName, functionArgs, availableFunctions):
        self.functionName = functionName
        self.functionArgs = functionArgs
        self.availableFunctions = availableFunctions

    def mainExecution(self):
        function_to_call = self.availableFunctions.get(self.functionName)
        if not function_to_call:
            return f"Error: Function '{self.functionName}' not found in available functions."
        # Check for required parameters
        import inspect
        sig = inspect.signature(function_to_call)
        missing = [p for p in sig.parameters if p not in self.functionArgs or self.functionArgs[p] in (None, "")]
        if missing:
            return f"Missing required parameter(s): {', '.join(missing)}. Please provide the value(s) to continue."
        return function_to_call(**self.functionArgs)