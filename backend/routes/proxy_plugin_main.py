from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services.proxy_plugin_main import ProxyPluginMainRequest
from backend.services.proxy_plugin_main import proxy_plugin_main_payload
from backend.settings import settings as _default_settings


class Utf8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


router = APIRouter()
settings = _default_settings


def bind_proxy_plugin_main_dependencies(*, settings_owner=None) -> None:
    global settings
    if settings_owner is not None:
        settings = settings_owner


@router.post("/proxy/plugin-main")
async def proxy_plugin_main(req: ProxyPluginMainRequest):
    result = await proxy_plugin_main_payload(req=req, settings=settings)
    return Utf8JSONResponse(status_code=result.status_code, content=result.content)
