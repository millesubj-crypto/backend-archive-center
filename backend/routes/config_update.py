import logging

from fastapi import APIRouter

from backend.services.runtime_config import update_runtime_config
from backend.settings import settings as _default_settings


router = APIRouter()
settings = _default_settings
critic = None
supervisor = None
logger = logging.getLogger(__name__)


def bind_runtime_config_dependencies(*, settings_owner=None, critic_owner=None, supervisor_owner=None, logger_owner=None) -> None:
    global settings, critic, supervisor, logger
    if settings_owner is not None:
        settings = settings_owner
    if critic_owner is not None:
        critic = critic_owner
    if supervisor_owner is not None:
        supervisor = supervisor_owner
    if logger_owner is not None:
        logger = logger_owner


@router.post("/config/update")
def config_update(body: dict):
    return update_runtime_config(
        body=body,
        settings=settings,
        critic=critic,
        supervisor=supervisor,
        logger=logger,
    )
