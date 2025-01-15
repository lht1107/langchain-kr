from langchain.callbacks.base import BaseCallbackHandler

class AnalysisStreamingHandler(BaseCallbackHandler):
    def __init__(self, analysis_type: str):
        self.analysis_type = analysis_type
        self.current_text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.current_text += token
        if token.endswith('.'):
            print(f"\n[{self.analysis_type}] {self.current_text.strip()}")
            self.current_text = ""