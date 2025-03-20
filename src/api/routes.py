from fastapi import FastAPI
import numpy as np
from datetime import datetime
import os

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
