
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Json


class DataContentOutputConfig(BaseModel):
    Format: Optional[str] = "pd"
    FilePath: Optional[str] = ""
