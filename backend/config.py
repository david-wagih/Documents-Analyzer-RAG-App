import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Get the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

class Settings:
    def __init__(self):
        # Use type assertion to handle Optional types
        self.OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
        self.TAVILY_API_KEY: str = os.getenv('TAVILY_API_KEY', '')
        self.ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    def validate(self) -> None:
        """Validate that required environment variables are set."""
        missing_vars = []
        
        if not self.OPENAI_API_KEY:
            missing_vars.append("OPENAI_API_KEY")
        if not self.TAVILY_API_KEY:
            missing_vars.append("TAVILY_API_KEY")
        if not self.ANTHROPIC_API_KEY:
            missing_vars.append("ANTHROPIC_API_KEY")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

settings = Settings() 