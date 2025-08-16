
# Run FastAPI (Uvicorn) on Windows
param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 5000
)
uvicorn backend.app:app --host $Host --port $Port --reload
