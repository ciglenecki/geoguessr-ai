import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve().parents[0]))

import os
import traceback
from glob import glob
from pathlib import Path

import uvicorn
from fastapi import FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import logger

from app.config import config
from app.routers import model_router
from app.server_store import ServerStore, server_store

app = FastAPI(debug=True)


async def catch_exceptions_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as err:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder({"message": "Internal server error. Check console log for more information."}),
        )


app.middleware("http")(catch_exceptions_middleware)


@app.on_event("shutdown")
def shutdown_event():
    with open("log.txt", mode="a") as log:
        log.write("Application shutdown")


@app.get("/")
async def root():
    logger.warning("Hello 43")

    return {"message": "Hello World"}


app.include_router(model_router.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
    )
