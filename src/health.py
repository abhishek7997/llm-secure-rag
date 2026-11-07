from config.database import connection_config
from config.llm_server import get_settings as llm_settings
import psycopg
import httpx
import logging

logger = logging.getLogger(__name__)

async def check_postgres() -> bool:
    try: 
        async with (await psycopg.AsyncConnection.connect(**connection_config)) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1;")
                await cur.fetchone()
            return True
    except psycopg.Error as e:
        logger.error(f"ERROR: COULD NOT CONNECT TO POSTGRES DB.\nError: {e}")
    
    return False

async def check_llm() -> bool:
    async with httpx.AsyncClient() as client:
        try:
            llm_response = await client.get(llm_settings().LLM_BASE_URL + llm_settings().LLM_MODEL_API, timeout=llm_settings().LLM_TIMEOUT)
            if llm_response.status_code == 200:
                return True
        except httpx.RequestError as e:
            logger.error(f"Unable to connect to LLM server. \nError: {e}")

    return False
