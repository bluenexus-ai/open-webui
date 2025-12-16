import time
import logging
import asyncio
import sys

from aiocache import cached
from fastapi import Request

from open_webui.routers import openai, ollama
from open_webui.functions import get_function_models


from open_webui.models.functions import Functions
from open_webui.models.models import Models


from open_webui.utils.plugin import (
    load_function_module_by_id,
    get_function_module_from_cache,
)
from open_webui.utils.access_control import has_access


from open_webui.config import (
    BYPASS_ADMIN_ACCESS_CONTROL,
    DEFAULT_ARENA_MODEL,
)

from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL, SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL

# Import BlueNexus session check - guarded to avoid circular imports
try:
    from open_webui.utils.bluenexus import has_bluenexus_session, is_bluenexus_enabled
    from open_webui.utils.bluenexus.config import is_bluenexus_data_storage_enabled
    from open_webui.repositories.factory import get_model_repository
except ImportError:
    has_bluenexus_session = lambda user_id: False
    is_bluenexus_enabled = lambda: False
    is_bluenexus_data_storage_enabled = lambda: False
    get_model_repository = None
from open_webui.models.users import UserModel


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def fetch_ollama_models(request: Request, user: UserModel = None):
    raw_ollama_models = await ollama.get_all_models(request, user=user)
    return [
        {
            "id": model["model"],
            "name": model["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama",
            "ollama": model,
            "connection_type": model.get("connection_type", "local"),
            "tags": model.get("tags", []),
        }
        for model in raw_ollama_models["models"]
    ]


async def fetch_openai_models(request: Request, user: UserModel = None):
    openai_response = await openai.get_all_models(request, user=user)
    return openai_response["data"]


async def get_all_base_models(request: Request, user: UserModel = None):
    openai_task = (
        fetch_openai_models(request, user)
        if request.app.state.config.ENABLE_OPENAI_API
        else asyncio.sleep(0, result=[])
    )
    ollama_task = (
        fetch_ollama_models(request, user)
        if request.app.state.config.ENABLE_OLLAMA_API
        else asyncio.sleep(0, result=[])
    )
    function_task = get_function_models(request)

    openai_models, ollama_models, function_models = await asyncio.gather(
        openai_task, ollama_task, function_task
    )

    return function_models + openai_models + ollama_models


async def get_all_models(request, refresh: bool = False, user: UserModel = None):
    if (
        request.app.state.MODELS
        and request.app.state.BASE_MODELS
        and (request.app.state.config.ENABLE_BASE_MODELS_CACHE and not refresh)
    ):
        base_models = request.app.state.BASE_MODELS
    else:
        base_models = await get_all_base_models(request, user=user)
        request.app.state.BASE_MODELS = base_models

    # deep copy the base models to avoid modifying the original list
    models = [model.copy() for model in base_models]

    # If there are no models, return an empty list
    if len(models) == 0:
        return []

    # Add arena models
    if request.app.state.config.ENABLE_EVALUATION_ARENA_MODELS:
        arena_models = []
        if len(request.app.state.config.EVALUATION_ARENA_MODELS) > 0:
            arena_models = [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "info": {
                        "meta": model["meta"],
                    },
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "arena",
                    "arena": True,
                }
                for model in request.app.state.config.EVALUATION_ARENA_MODELS
            ]
        else:
            # Add default arena model
            arena_models = [
                {
                    "id": DEFAULT_ARENA_MODEL["id"],
                    "name": DEFAULT_ARENA_MODEL["name"],
                    "info": {
                        "meta": DEFAULT_ARENA_MODEL["meta"],
                    },
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "arena",
                    "arena": True,
                }
            ]
        models = models + arena_models

    global_action_ids = [
        function.id for function in Functions.get_global_action_functions()
    ]
    enabled_action_ids = [
        function.id
        for function in Functions.get_functions_by_type("action", active_only=True)
    ]

    global_filter_ids = [
        function.id for function in Functions.get_global_filter_functions()
    ]
    enabled_filter_ids = [
        function.id
        for function in Functions.get_functions_by_type("filter", active_only=True)
    ]

    # Fetch custom models - use BlueNexus OR PostgreSQL based on config (not both)
    custom_models = []
    bluenexus_storage_enabled = is_bluenexus_data_storage_enabled()

    if bluenexus_storage_enabled and get_model_repository and user:
        # ONLY use BlueNexus when BLUENEXUS_DATA_STORAGE=true
        try:
            repo = get_model_repository(user.id)
            bn_models = await repo.get_list(user.id)
            # Wrap dicts in SimpleNamespace for attribute access, add model_dump method
            from types import SimpleNamespace
            for m in bn_models:
                ns = SimpleNamespace(**m)
                ns.model_dump = lambda _m=m: _m
                # Handle meta as SimpleNamespace too if it's a dict
                if isinstance(m.get("meta"), dict):
                    ns.meta = SimpleNamespace(**m["meta"])
                    ns.meta.model_dump = lambda _meta=m["meta"]: _meta
                custom_models.append(ns)
            log.info(f"[get_all_models] Loaded {len(custom_models)} models from BlueNexus (storage=true)")
        except Exception as e:
            log.warning(f"[get_all_models] Failed to load BlueNexus models: {e}")
    else:
        # ONLY use PostgreSQL when BLUENEXUS_DATA_STORAGE=false
        custom_models = list(Models.get_all_models())
        log.info(f"[get_all_models] Loaded {len(custom_models)} models from PostgreSQL (storage=false)")

    for custom_model in custom_models:
        log.info(f"[get_all_models] Processing custom model: id={custom_model.id}, name={getattr(custom_model, 'name', 'N/A')}, base_model_id={getattr(custom_model, 'base_model_id', 'N/A')}, is_active={getattr(custom_model, 'is_active', 'N/A')}")
        if custom_model.base_model_id is None:
            # Applied directly to a base model
            matched_base_model = False
            for model in models:
                if custom_model.id == model["id"] or (
                    model.get("owned_by") == "ollama"
                    and custom_model.id
                    == model["id"].split(":")[
                        0
                    ]  # Ollama may return model ids in different formats (e.g., 'llama3' vs. 'llama3:7b')
                ):
                    matched_base_model = True
                    if custom_model.is_active:
                        model["name"] = custom_model.name
                        model["info"] = custom_model.model_dump()

                        # Set action_ids and filter_ids
                        action_ids = []
                        filter_ids = []

                        if "info" in model:
                            if "meta" in model["info"]:
                                action_ids.extend(
                                    model["info"]["meta"].get("actionIds", [])
                                )
                                filter_ids.extend(
                                    model["info"]["meta"].get("filterIds", [])
                                )

                            if "params" in model["info"]:
                                # Remove params to avoid exposing sensitive info
                                del model["info"]["params"]

                        model["action_ids"] = action_ids
                        model["filter_ids"] = filter_ids
                    else:
                        models.remove(model)
                    break

            # If no base model matched and model is active, add as standalone custom model
            if not matched_base_model and custom_model.is_active and (
                custom_model.id not in [model["id"] for model in models]
            ):
                model = {
                    "id": f"{custom_model.id}",
                    "name": custom_model.name,
                    "object": "model",
                    "created": getattr(custom_model, 'created_at', int(time.time())),
                    "owned_by": "custom",
                    "preset": True,
                }

                info = custom_model.model_dump()
                if "params" in info:
                    del info["params"]
                model["info"] = info

                action_ids = []
                filter_ids = []
                if hasattr(custom_model, 'meta') and custom_model.meta:
                    meta = custom_model.meta.model_dump() if hasattr(custom_model.meta, 'model_dump') else custom_model.meta
                    if isinstance(meta, dict):
                        action_ids.extend(meta.get("actionIds", []))
                        filter_ids.extend(meta.get("filterIds", []))

                model["action_ids"] = action_ids
                model["filter_ids"] = filter_ids
                models.append(model)
                log.info(f"[get_all_models] Added standalone custom model: {custom_model.id}")

        elif custom_model.is_active and (
            custom_model.id not in [model["id"] for model in models]
        ):
            # Custom model based on a base model
            try:
                owned_by = "openai"
                pipe = None

                for m in models:
                    if (
                        custom_model.base_model_id == m["id"]
                        or custom_model.base_model_id == m["id"].split(":")[0]
                    ):
                        owned_by = m.get("owned_by", "unknown")
                        if "pipe" in m:
                            pipe = m["pipe"]
                        break

                model = {
                    "id": f"{custom_model.id}",
                    "name": custom_model.name,
                    "object": "model",
                    "created": getattr(custom_model, 'created_at', int(time.time())),
                    "owned_by": owned_by,
                    "preset": True,
                    **({"pipe": pipe} if pipe is not None else {}),
                }

                info = custom_model.model_dump()
                if "params" in info:
                    # Remove params to avoid exposing sensitive info
                    del info["params"]

                model["info"] = info

                action_ids = []
                filter_ids = []

                if hasattr(custom_model, 'meta') and custom_model.meta:
                    meta = custom_model.meta.model_dump() if hasattr(custom_model.meta, 'model_dump') else (custom_model.meta if isinstance(custom_model.meta, dict) else {})

                    if isinstance(meta, dict):
                        if "actionIds" in meta:
                            action_ids.extend(meta["actionIds"])

                        if "filterIds" in meta:
                            filter_ids.extend(meta["filterIds"])

                model["action_ids"] = action_ids
                model["filter_ids"] = filter_ids

                models.append(model)
                log.info(f"[get_all_models] Added preset custom model: id={custom_model.id}, name={custom_model.name}")
            except Exception as e:
                log.error(f"[get_all_models] Failed to add preset custom model {custom_model.id}: {e}")

    # Process action_ids to get the actions
    def get_action_items_from_module(function, module):
        actions = []
        if hasattr(module, "actions"):
            actions = module.actions
            return [
                {
                    "id": f"{function.id}.{action['id']}",
                    "name": action.get("name", f"{function.name} ({action['id']})"),
                    "description": function.meta.description,
                    "icon": action.get(
                        "icon_url",
                        function.meta.manifest.get("icon_url", None)
                        or getattr(module, "icon_url", None)
                        or getattr(module, "icon", None),
                    ),
                }
                for action in actions
            ]
        else:
            return [
                {
                    "id": function.id,
                    "name": function.name,
                    "description": function.meta.description,
                    "icon": function.meta.manifest.get("icon_url", None)
                    or getattr(module, "icon_url", None)
                    or getattr(module, "icon", None),
                }
            ]

    # Process filter_ids to get the filters
    def get_filter_items_from_module(function, module):
        return [
            {
                "id": function.id,
                "name": function.name,
                "description": function.meta.description,
                "icon": function.meta.manifest.get("icon_url", None)
                or getattr(module, "icon_url", None)
                or getattr(module, "icon", None),
                "has_user_valves": hasattr(module, "UserValves"),
            }
        ]

    def get_function_module_by_id(function_id):
        function_module, _, _ = get_function_module_from_cache(request, function_id)
        return function_module

    for model in models:
        action_ids = [
            action_id
            for action_id in list(set(model.pop("action_ids", []) + global_action_ids))
            if action_id in enabled_action_ids
        ]
        filter_ids = [
            filter_id
            for filter_id in list(set(model.pop("filter_ids", []) + global_filter_ids))
            if filter_id in enabled_filter_ids
        ]

        model["actions"] = []
        for action_id in action_ids:
            action_function = Functions.get_function_by_id(action_id)
            if action_function is None:
                raise Exception(f"Action not found: {action_id}")

            function_module = get_function_module_by_id(action_id)
            model["actions"].extend(
                get_action_items_from_module(action_function, function_module)
            )

        model["filters"] = []
        for filter_id in filter_ids:
            filter_function = Functions.get_function_by_id(filter_id)
            if filter_function is None:
                raise Exception(f"Filter not found: {filter_id}")

            function_module = get_function_module_by_id(filter_id)

            if getattr(function_module, "toggle", None):
                model["filters"].extend(
                    get_filter_items_from_module(filter_function, function_module)
                )

    log.debug(f"get_all_models() returned {len(models)} models")

    request.app.state.MODELS = {model["id"]: model for model in models}
    return models


def check_model_access(user, model):
    model_id = model.get("id", "unknown")
    log.info(f"[check_model_access] model_id={model_id}, user={user.id}, role={user.role}, tags={model.get('tags', [])}")

    if model.get("arena"):
        if not has_access(
            user.id,
            type="read",
            access_control=model.get("info", {})
            .get("meta", {})
            .get("access_control", {}),
        ):
            raise Exception("Model not found")
    else:
        # Check if this is a BlueNexus model and user has BlueNexus session
        model_tags = model.get("tags", [])
        tag_names = []
        for tag in model_tags:
            if isinstance(tag, str):
                tag_names.append(tag)
            elif isinstance(tag, dict) and "name" in tag:
                tag_names.append(tag.get("name"))

        log.info(f"[check_model_access] tag_names={tag_names}")

        if "bluenexus" in tag_names:
            # Allow BlueNexus models for users with BlueNexus session
            bn_enabled = is_bluenexus_enabled()
            bn_session = has_bluenexus_session(user.id) if bn_enabled else False
            log.info(f"[check_model_access] BlueNexus model detected: enabled={bn_enabled}, has_session={bn_session}")
            if bn_enabled and bn_session:
                return  # Access granted
            else:
                raise Exception("Model not found")

        model_info = Models.get_model_by_id(model.get("id"))
        if not model_info:
            log.info(f"[check_model_access] Model not in database: {model_id}")
            raise Exception("Model not found")
        elif not (
            user.id == model_info.user_id
            or has_access(
                user.id, type="read", access_control=model_info.access_control
            )
        ):
            raise Exception("Model not found")


async def get_filtered_models(models, user):
    # Filter out models that the user does not have access to
    if (
        user.role == "user"
        or (user.role == "admin" and not BYPASS_ADMIN_ACCESS_CONTROL)
    ) and not BYPASS_MODEL_ACCESS_CONTROL:
        filtered_models = []

        # Check if user has BlueNexus session (cached for this filtering pass)
        bluenexus_enabled = is_bluenexus_enabled()
        user_has_session = has_bluenexus_session(user.id) if bluenexus_enabled else False
        user_has_bluenexus = bluenexus_enabled and user_has_session

        # Check if BlueNexus data storage is enabled
        bluenexus_data_storage = is_bluenexus_data_storage_enabled()

        log.info(f"[get_filtered_models] user={user.id}, role={user.role}, bluenexus_enabled={bluenexus_enabled}, has_session={user_has_session}, data_storage={bluenexus_data_storage}")

        for model in models:
            model_id = model.get("id", "unknown")

            if model.get("arena"):
                if has_access(
                    user.id,
                    type="read",
                    access_control=model.get("info", {})
                    .get("meta", {})
                    .get("access_control", {}),
                ):
                    filtered_models.append(model)
                continue

            # Allow BlueNexus models for users with a valid BlueNexus session
            model_tags = model.get("tags", [])
            # Handle tags that could be strings or dicts with "name" key
            tag_names = []
            for tag in model_tags:
                if isinstance(tag, str):
                    tag_names.append(tag)
                elif isinstance(tag, dict) and "name" in tag:
                    tag_names.append(tag["name"])

            # Debug: log model tags for first few models
            if len(filtered_models) < 3:
                log.info(f"[get_filtered_models] model={model_id}, tags={model_tags}, tag_names={tag_names}")

            if "bluenexus" in tag_names and user_has_bluenexus:
                filtered_models.append(model)
                continue

            # For preset models, use the info already attached from get_all_models
            # This avoids a separate lookup that may fail due to ID mismatch
            model_info = model.get("info")
            if model_info:
                log.info(f"[get_filtered_models] Using attached info for model {model_id}, user_id={model_info.get('user_id')}, current_user={user.id}")
            else:
                # Check model access - use BlueNexus or PostgreSQL based on config
                if bluenexus_data_storage and get_model_repository:
                    try:
                        repo = get_model_repository(user.id)
                        model_info = await repo.get_by_id(model["id"])
                    except Exception as e:
                        log.debug(f"[get_filtered_models] BlueNexus model lookup failed: {e}")
                        model_info = None

                # Fallback to PostgreSQL
                if model_info is None:
                    model_info = Models.get_model_by_id(model["id"])
                    if model_info:
                        model_info = model_info.model_dump()

            if model_info:
                # Check access: admin bypass, owner, or has read access
                is_admin_bypass = user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL
                info_user_id = model_info.get("user_id") if isinstance(model_info, dict) else getattr(model_info, 'user_id', None)
                is_owner = user.id == info_user_id
                access_control = model_info.get("access_control") if isinstance(model_info, dict) else getattr(model_info, 'access_control', None)
                has_read_access = has_access(user.id, type="read", access_control=access_control)

                log.info(f"[get_filtered_models] Access check for {model_id}: is_admin_bypass={is_admin_bypass}, is_owner={is_owner} (user={user.id}, info_user_id={info_user_id}), has_read_access={has_read_access}")

                if is_admin_bypass or is_owner or has_read_access:
                    filtered_models.append(model)
                    log.info(f"[get_filtered_models] Model {model_id} PASSED access check")

        return filtered_models
    else:
        return models
