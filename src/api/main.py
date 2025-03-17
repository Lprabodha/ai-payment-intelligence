from fastapi import FastAPI
from .routes import router

app = FastAPI(title="AI Payment Intelligence API")

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
