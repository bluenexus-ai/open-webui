from typing import Optional
import io
import base64
import logging
import time

from open_webui.models.models import (
    ModelForm,
    ModelModel,
    ModelResponse,
    ModelUserResponse,
    ModelParams,
    ModelMeta,
    Models,
)
from open_webui.models.users import Users

from pydantic import BaseModel
from open_webui.constants import ERROR_MESSAGES
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
    Response,
)
from fastapi.responses import FileResponse, StreamingResponse


from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.utils.bluenexus.config import is_bluenexus_data_storage_enabled
from open_webui.repositories import get_model_repository
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL, STATIC_DIR

log = logging.getLogger(__name__)

router = APIRouter()


def validate_model_id(model_id: str) -> bool:
    return model_id and len(model_id) <= 256


def _ensure_model_defaults(model_data: dict) -> dict:
    """Ensure model data has required default values."""
    # Ensure params has default value
    if "params" not in model_data or model_data["params"] is None:
        model_data["params"] = ModelParams().model_dump()

    # Ensure meta has default value
    if "meta" not in model_data or model_data["meta"] is None:
        model_data["meta"] = ModelMeta(description="").model_dump()

    # Ensure is_active has default
    if "is_active" not in model_data:
        model_data["is_active"] = True

    # Ensure timestamps exist
    if "created_at" not in model_data:
        model_data["created_at"] = int(time.time_ns())
    if "updated_at" not in model_data:
        model_data["updated_at"] = int(time.time_ns())

    return model_data


###########################
# GetModels
###########################


@router.get(
    "/list", response_model=list[ModelUserResponse]
)  # do NOT use "/" as path, conflicts with main.py
async def get_models(id: Optional[str] = None, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        models_data = await repo.get_list(user.id)

        # Collect all unique user_ids and batch fetch users
        user_ids = list(set(m.get("user_id") for m in models_data if m.get("user_id")))
        users = Users.get_users_by_user_ids(user_ids) if user_ids else []
        users_dict = {u.id: u for u in users}

        models = []
        for model_data in models_data:
            model_data = _ensure_model_defaults(model_data)

            # Get user info for this model
            model_user = users_dict.get(model_data.get("user_id"))
            model_data["user"] = model_user.model_dump() if model_user else None

            # Apply access control - user always has access to their own models
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                models.append(ModelUserResponse(**model_data))
            elif model_data.get("user_id") == user.id:
                models.append(ModelUserResponse(**model_data))
            elif has_access(user.id, "read", model_data.get("access_control")):
                models.append(ModelUserResponse(**model_data))

        return models
    else:
        # PostgreSQL path
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            models = Models.get_all_models()
        else:
            models = Models.get_models_by_user_id(user.id)

        user_ids = list(set(m.user_id for m in models if m.user_id))
        users = Users.get_users_by_user_ids(user_ids) if user_ids else []
        users_dict = {u.id: u for u in users}

        result = []
        for model in models:
            model_user = users_dict.get(model.user_id)
            model_data = model.model_dump()
            model_data["user"] = model_user.model_dump() if model_user else None
            result.append(ModelUserResponse(**model_data))

        return result


###########################
# GetBaseModels
###########################


@router.get("/base", response_model=list[ModelResponse])
async def get_base_models(user=Depends(get_admin_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        models_data = await repo.get_all()

        base_models = []
        for model_data in models_data:
            model_data = _ensure_model_defaults(model_data)
            # Filter for base models (those with base_model_id set)
            if model_data.get("base_model_id"):
                base_models.append(ModelResponse(**model_data))

        return base_models
    else:
        # PostgreSQL path
        models = Models.get_base_models()
        return [ModelResponse(**m.model_dump()) for m in models]


############################
# CreateNewModel
############################


@router.post("/create", response_model=Optional[ModelModel])
async def create_new_model(
    request: Request,
    form_data: ModelForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.models", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not validate_model_id(form_data.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.MODEL_ID_TOO_LONG,
        )

    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)

        # Check if model with this ID already exists
        existing = await repo.get_by_id(form_data.id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.MODEL_ID_TAKEN,
            )

        # Create new model
        model_data = form_data.model_dump()
        result = await repo.create(user.id, model_data)
        result = _ensure_model_defaults(result)
        return ModelModel(**result) if result else None
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(form_data.id)
        if model:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.MODEL_ID_TAKEN,
            )

        model = Models.insert_new_model(form_data, user.id)
        return model


############################
# ExportModels
############################


@router.get("/export", response_model=list[ModelModel])
async def export_models(user=Depends(get_admin_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        models_data = await repo.get_all()

        models = []
        for model_data in models_data:
            model_data = _ensure_model_defaults(model_data)
            models.append(ModelModel(**model_data))

        return models
    else:
        # PostgreSQL path
        return Models.get_all_models()


############################
# ImportModels
############################


class ModelsImportForm(BaseModel):
    models: list[dict]


@router.post("/import", response_model=bool)
async def import_models(
    user=Depends(get_admin_user), form_data: ModelsImportForm = (...)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        data = form_data.models

        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        for model_data in data:
            model_id = model_data.get("id")

            if not model_id or not validate_model_id(model_id):
                continue

            # Check if model already exists
            existing = await repo.get_by_id(model_id)

            model_data["meta"] = model_data.get("meta", {})
            model_data["params"] = model_data.get("params", {})

            if existing:
                # Update existing model
                await repo.update(model_id, model_data)
            else:
                # Insert new model
                await repo.create(user.id, model_data)

        return True
    else:
        # PostgreSQL path
        data = form_data.models
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        for model_data in data:
            model_id = model_data.get("id")
            if not model_id or not validate_model_id(model_id):
                continue

            model_data["meta"] = model_data.get("meta", {})
            model_data["params"] = model_data.get("params", {})

            existing = Models.get_model_by_id(model_id)
            if existing:
                form = ModelForm(**model_data)
                Models.update_model_by_id(model_id, form)
            else:
                form = ModelForm(**model_data)
                Models.insert_new_model(form, user.id)

        return True


############################
# SyncModels
############################


class SyncModelsForm(BaseModel):
    models: list[ModelModel] = []


@router.post("/sync", response_model=list[ModelModel])
async def sync_models(
    request: Request, form_data: SyncModelsForm, user=Depends(get_admin_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        synced_models = []

        for model in form_data.models:
            model_data = model.model_dump()
            model_id = model_data.get("id")

            if not model_id:
                continue

            # Check if model exists
            existing = await repo.get_by_id(model_id)

            if existing:
                # Update existing model
                result = await repo.update(model_id, model_data)
            else:
                # Create new model
                result = await repo.create(user.id, model_data)

            if result:
                result = _ensure_model_defaults(result)
                synced_models.append(ModelModel(**result))

        return synced_models
    else:
        # PostgreSQL path
        synced_models = []

        for model in form_data.models:
            model_data = model.model_dump()
            model_id = model_data.get("id")

            if not model_id:
                continue

            existing = Models.get_model_by_id(model_id)
            form = ModelForm(**model_data)

            if existing:
                result = Models.update_model_by_id(model_id, form)
            else:
                result = Models.insert_new_model(form, user.id)

            if result:
                synced_models.append(result)

        return synced_models


###########################
# GetModelById
###########################


# Note: We're not using the typical url path param here, but instead using a query parameter to allow '/' in the id
@router.get("/model", response_model=Optional[ModelResponse])
async def get_model_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        model_data = await repo.get_by_id(id)

        if not model_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        model_data = _ensure_model_defaults(model_data)

        # Check access
        if (
            (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
            or model_data.get("user_id") == user.id
            or has_access(user.id, "read", model_data.get("access_control"))
        ):
            return ModelResponse(**model_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
            or model.user_id == user.id
            or has_access(user.id, "read", model.access_control)
        ):
            return ModelResponse(**model.model_dump())
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )


###########################
# GetModelById
###########################


@router.get("/model/profile/image")
async def get_model_profile_image(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        model_data = await repo.get_by_id(id)

        if model_data:
            meta = model_data.get("meta", {})
            profile_image_url = meta.get("profile_image_url") if isinstance(meta, dict) else None

            if profile_image_url:
                if profile_image_url.startswith("http"):
                    return Response(
                        status_code=status.HTTP_302_FOUND,
                        headers={"Location": profile_image_url},
                    )
                elif profile_image_url.startswith("data:image"):
                    try:
                        header, base64_data = profile_image_url.split(",", 1)
                        image_data = base64.b64decode(base64_data)
                        image_buffer = io.BytesIO(image_data)

                        return StreamingResponse(
                            image_buffer,
                            media_type="image/png",
                            headers={"Content-Disposition": "inline; filename=image.png"},
                        )
                    except Exception:
                        pass

        return FileResponse(f"{STATIC_DIR}/favicon.png")
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(id)
        if model and model.meta:
            profile_image_url = model.meta.profile_image_url if hasattr(model.meta, 'profile_image_url') else None
            if not profile_image_url and isinstance(model.meta, dict):
                profile_image_url = model.meta.get("profile_image_url")

            if profile_image_url:
                if profile_image_url.startswith("http"):
                    return Response(
                        status_code=status.HTTP_302_FOUND,
                        headers={"Location": profile_image_url},
                    )
                elif profile_image_url.startswith("data:image"):
                    try:
                        header, base64_data = profile_image_url.split(",", 1)
                        image_data = base64.b64decode(base64_data)
                        image_buffer = io.BytesIO(image_data)

                        return StreamingResponse(
                            image_buffer,
                            media_type="image/png",
                            headers={"Content-Disposition": "inline; filename=image.png"},
                        )
                    except Exception:
                        pass

        return FileResponse(f"{STATIC_DIR}/favicon.png")


############################
# ToggleModelById
############################


@router.post("/model/toggle", response_model=Optional[ModelResponse])
async def toggle_model_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        model_data = await repo.get_by_id(id)

        if not model_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        # Check access
        if not (
            user.role == "admin"
            or model_data.get("user_id") == user.id
            or has_access(user.id, "write", model_data.get("access_control"))
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        # Toggle is_active field
        model_data["is_active"] = not model_data.get("is_active", True)

        result = await repo.update(id, model_data)
        if result:
            result = _ensure_model_defaults(result)
            return ModelResponse(**result)
        return None
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if not (
            user.role == "admin"
            or model.user_id == user.id
            or has_access(user.id, "write", model.access_control)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        model = Models.toggle_model_active_by_id(id)
        if model:
            return ModelResponse(**model.model_dump())
        return None


############################
# UpdateModelById
############################


@router.post("/model/update", response_model=Optional[ModelModel])
async def update_model_by_id(
    id: str,
    form_data: ModelForm,
    user=Depends(get_verified_user),
):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        model_data = await repo.get_by_id(id)

        if not model_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        # Check access
        if (
            model_data.get("user_id") != user.id
            and not has_access(user.id, "write", model_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Update model
        update_data = form_data.model_dump()
        result = await repo.update(id, update_data)
        if result:
            result = _ensure_model_defaults(result)
            return ModelModel(**result)
        return None
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            model.user_id != user.id
            and not has_access(user.id, "write", model.access_control)
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        model = Models.update_model_by_id(id, form_data)
        return model


############################
# DeleteModelById
############################


@router.delete("/model/delete", response_model=bool)
async def delete_model_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)
        model_data = await repo.get_by_id(id)

        if not model_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        # Check access
        if (
            user.role != "admin"
            and model_data.get("user_id") != user.id
            and not has_access(user.id, "write", model_data.get("access_control"))
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        return await repo.delete(id)
    else:
        # PostgreSQL path
        model = Models.get_model_by_id(id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            user.role != "admin"
            and model.user_id != user.id
            and not has_access(user.id, "write", model.access_control)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        return Models.delete_model_by_id(id)


@router.delete("/delete/all", response_model=bool)
async def delete_all_models(user=Depends(get_admin_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_model_repository(user.id)

        # Get all models and delete them
        models_data = await repo.get_all()
        for model_data in models_data:
            model_id = model_data.get("id")
            if model_id:
                await repo.delete(model_id)

        return True
    else:
        # PostgreSQL path
        return Models.delete_all_models()
