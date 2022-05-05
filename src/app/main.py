import sys
from pathlib import Path

# include project path so that .env file can be read from all locations
sys.path.append(str(Path(__file__).parent.resolve().parents[0]))

import json
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.config import config
from app.routers import router
from logger import logger
from descriptions import predict_desc, api_description


def catch_exceptions_middleware(request, call_next):
    try:
        return call_next(request)
    except Exception as err:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder({"message": "Internal server error. Check console log for more information."}),
        )


tags_metadata = [
    {
        "name": "available models",
        "description": "All available models from the `MODEL_DIRECTORY` directory defined in the `.env` file",
    },
    {
        "name": "predict",
        "description": predict_desc,
    },
]

app = FastAPI(debug=True, openapi_tags=tags_metadata, description=api_description)
app.middleware("http")(catch_exceptions_middleware)


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown")


app.include_router(router.router)

with open(Path(Path(__file__).parent.resolve(), "openapi_spec.json"), "w+") as file:
    file.write(json.dumps(app.openapi()))
    file.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["HOST"],
        port=int(config["PORT"]),
        loop="asyncio",
        reload=bool(int(config["HOT_RELOAD"])),
    )
