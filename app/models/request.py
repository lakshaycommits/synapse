from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")

class SyncRequest(BaseModel):
    repo: str = Field(..., description="GitHub repository in owner/repo format")
    branch: str = Field("main", description="Branch to sync code files from")
