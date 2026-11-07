from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from config.database import get_settings as db_settings
from config.llm_server import get_settings as llm_settings
from health import check_postgres, check_llm
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_settings()
    llm_settings()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health/ready")
async def ready(response: Response):
    postgres_live, llm_live = await asyncio.gather(check_postgres(), check_llm())

    if postgres_live and llm_live:
        return { "status": "success"}
    
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "unavailable", 
        "checks": {
            "postgres": "ok" if postgres_live else "unreachable", 
            "llm": "ok" if llm_live else "unreachable"
        }
    }

@app.get("/health/live")
async def live():
    return {"status":"live"}
