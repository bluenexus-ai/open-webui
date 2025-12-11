import logging
from pathlib import Path
from typing import Optional
import time
import re
import aiohttp
from open_webui.models.groups import Groups
from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, Depends, HTTPException, Request, status


from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.models.tools import (
    ToolForm,
    ToolModel,
    ToolResponse,
    ToolUserResponse,
)
from open_webui.utils.plugin import (
    load_tool_module_by_id,
    replace_imports,
    get_tool_module_from_cache,
)
from open_webui.utils.tools import get_tool_specs
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.utils.tools import get_tool_servers
from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions, BlueNexusError

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import CACHE_DIR, BYPASS_ADMIN_ACCESS_CONTROL
from open_webui.constants import ERROR_MESSAGES


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


router = APIRouter()


def get_client_or_raise(user_id: str):
    """Get BlueNexus client for user or raise 401 error."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="BlueNexus session required. Please log in with BlueNexus.",
        )
    return client


def get_tool_module(request, tool_id, load_from_db=True, user_id=None):
    """
    Get the tool module by its ID.
    """
    tool_module, _ = get_tool_module_from_cache(request, tool_id, load_from_db, user_id=user_id)
    return tool_module


############################
# GetTools
############################


@router.get("/", response_model=list[ToolUserResponse])
async def get_tools(request: Request, user=Depends(get_verified_user)):
    tools = []

    # Local Tools from BlueNexus
    client = get_bluenexus_client_for_user(user.id)
    if client:
        try:
            response = await client.query(
                Collections.TOOLS,
                QueryOptions(filter={"user_id": user.id}, limit=100)
            )
            for record in response.get_records():
                tool_data = record.model_dump()
                tool_id = tool_data.get("owui_id", tool_data.get("id"))
                tool_data["id"] = tool_id
                tool_module = get_tool_module(request, tool_id, load_from_db=False, user_id=user.id)
                tools.append(
                    ToolUserResponse(
                        **{
                            **tool_data,
                            "has_user_valves": hasattr(tool_module, "UserValves") if tool_module else False,
                        }
                    )
                )
        except BlueNexusError as e:
            log.warning(f"Failed to fetch tools from BlueNexus: {e}")

    # OpenAPI Tool Servers
    for server in await get_tool_servers(request):
        tools.append(
            ToolUserResponse(
                **{
                    "id": f"server:{server.get('id')}",
                    "user_id": f"server:{server.get('id')}",
                    "name": server.get("openapi", {})
                    .get("info", {})
                    .get("title", "Tool Server"),
                    "meta": {
                        "description": server.get("openapi", {})
                        .get("info", {})
                        .get("description", ""),
                    },
                    "access_control": request.app.state.config.TOOL_SERVER_CONNECTIONS[
                        server.get("idx", 0)
                    ]
                    .get("config", {})
                    .get("access_control", None),
                    "updated_at": int(time.time()),
                    "created_at": int(time.time()),
                }
            )
        )

    # MCP Tool Servers
    for server in request.app.state.config.TOOL_SERVER_CONNECTIONS:
        if server.get("type", "openapi") == "mcp":
            server_id = server.get("info", {}).get("id")
            auth_type = server.get("auth_type", "none")

            session_token = None
            if auth_type == "oauth_2.1":
                splits = server_id.split(":")
                server_id = splits[-1] if len(splits) > 1 else server_id

                session_token = (
                    await request.app.state.oauth_client_manager.get_oauth_token(
                        user.id, f"mcp:{server_id}"
                    )
                )

            tools.append(
                ToolUserResponse(
                    **{
                        "id": f"server:mcp:{server.get('info', {}).get('id')}",
                        "user_id": f"server:mcp:{server.get('info', {}).get('id')}",
                        "name": server.get("info", {}).get("name", "MCP Tool Server"),
                        "meta": {
                            "description": server.get("info", {}).get(
                                "description", ""
                            ),
                        },
                        "access_control": server.get("config", {}).get(
                            "access_control", None
                        ),
                        "updated_at": int(time.time()),
                        "created_at": int(time.time()),
                        **(
                            {
                                "authenticated": session_token is not None,
                            }
                            if auth_type == "oauth_2.1"
                            else {}
                        ),
                    }
                )
            )

    if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
        # Admin can see all tools
        return tools
    else:
        user_group_ids = {group.id for group in Groups.get_groups_by_member_id(user.id)}
        tools = [
            tool
            for tool in tools
            if tool.user_id == user.id
            or has_access(user.id, "read", tool.access_control, user_group_ids)
        ]
        return tools


############################
# GetToolList
############################


@router.get("/list", response_model=list[ToolUserResponse])
async def get_tool_list(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all tools from BlueNexus for this user
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"user_id": user.id})
        )

        records = response.get_records()
        tools = []

        for record in records:
            tool_data = record.model_dump()
            tool_data["id"] = tool_data.get("owui_id", tool_data.get("id"))

            # Apply access control for write permission
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                tools.append(ToolUserResponse(**tool_data))
            elif has_access(user.id, "write", tool_data.get("access_control")):
                tools.append(ToolUserResponse(**tool_data))

        return tools

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# LoadFunctionFromLink
############################


class LoadUrlForm(BaseModel):
    url: HttpUrl


def github_url_to_raw_url(url: str) -> str:
    # Handle 'tree' (folder) URLs (add main.py at the end)
    m1 = re.match(r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)", url)
    if m1:
        org, repo, branch, path = m1.groups()
        return f"https://raw.githubusercontent.com/{org}/{repo}/refs/heads/{branch}/{path.rstrip('/')}/main.py"

    # Handle 'blob' (file) URLs
    m2 = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m2:
        org, repo, branch, path = m2.groups()
        return (
            f"https://raw.githubusercontent.com/{org}/{repo}/refs/heads/{branch}/{path}"
        )

    # No match; return as-is
    return url


@router.post("/load/url", response_model=Optional[dict])
async def load_tool_from_url(
    request: Request, form_data: LoadUrlForm, user=Depends(get_admin_user)
):
    # NOTE: This is NOT a SSRF vulnerability:
    # This endpoint is admin-only (see get_admin_user), meant for *trusted* internal use,
    # and does NOT accept untrusted user input. Access is enforced by authentication.

    url = str(form_data.url)
    if not url:
        raise HTTPException(status_code=400, detail="Please enter a valid URL")

    url = github_url_to_raw_url(url)
    url_parts = url.rstrip("/").split("/")

    file_name = url_parts[-1]
    tool_name = (
        file_name[:-3]
        if (
            file_name.endswith(".py")
            and (not file_name.startswith(("main.py", "index.py", "__init__.py")))
        )
        else url_parts[-2] if len(url_parts) > 1 else "function"
    )

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                url, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=resp.status, detail="Failed to fetch the tool"
                    )
                data = await resp.text()
                if not data:
                    raise HTTPException(
                        status_code=400, detail="No data received from the URL"
                    )
        return {
            "name": tool_name,
            "content": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing tool: {e}")


############################
# ExportTools
############################


@router.get("/export", response_model=list[ToolModel])
async def export_tools(user=Depends(get_admin_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all tools from BlueNexus for export
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"user_id": user.id})
        )

        tools = []
        for record in response.get_records():
            tool_data = record.model_dump()
            tool_data["id"] = tool_data.get("owui_id", tool_data.get("id"))
            tools.append(ToolModel(**tool_data))

        return tools

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CreateNewTools
############################


@router.post("/create", response_model=Optional[ToolResponse])
async def create_new_tools(
    request: Request,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.tools", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not form_data.id.isidentifier():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only alphanumeric characters and underscores are allowed in the id",
        )

    form_data.id = form_data.id.lower()
    client = get_client_or_raise(user.id)

    try:
        # Check if tool with this ID already exists
        existing = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": form_data.id, "user_id": user.id}, limit=1)
        )

        if existing.data and len(existing.data) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ID_TAKEN,
            )

        # Process tool module
        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = load_tool_module_by_id(
            form_data.id, content=form_data.content
        )
        form_data.meta.manifest = frontmatter

        TOOLS = request.app.state.TOOLS
        TOOLS[form_data.id] = tool_module

        specs = get_tool_specs(TOOLS[form_data.id])

        tool_cache_dir = CACHE_DIR / "tools" / form_data.id
        tool_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create tool in BlueNexus
        tool_data = form_data.model_dump()
        tool_data["owui_id"] = form_data.id
        tool_data["user_id"] = user.id
        tool_data["specs"] = specs

        record = await client.create(Collections.TOOLS, tool_data)

        # Convert back to ToolResponse
        result_data = record.model_dump()
        result_data["id"] = result_data.get("owui_id", result_data.get("id"))
        return ToolResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        log.exception(f"Failed to load the tool by id {form_data.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# GetToolsById
############################


@router.get("/id/{id}", response_model=Optional[ToolModel])
async def get_tools_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query BlueNexus for tool with this ID
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        tool_data = record.model_dump()
        tool_data["id"] = tool_data.get("owui_id", tool_data.get("id"))

        # Check access
        if (
            user.role == "admin"
            or tool_data.get("user_id") == user.id
            or has_access(user.id, "read", tool_data.get("access_control"))
        ):
            return ToolModel(**tool_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateToolsById
############################


@router.post("/id/{id}/update", response_model=Optional[ToolModel])
async def update_tools_by_id(
    request: Request,
    id: str,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    client = get_client_or_raise(user.id)

    try:
        # Find the tool by ID
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        tool_data = record.model_dump()

        # Check access
        if (
            tool_data.get("user_id") != user.id
            and not has_access(user.id, "write", tool_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        # Process tool module
        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = load_tool_module_by_id(id, content=form_data.content)
        form_data.meta.manifest = frontmatter

        TOOLS = request.app.state.TOOLS
        TOOLS[id] = tool_module

        specs = get_tool_specs(TOOLS[id])

        # Update in BlueNexus
        updated_data = form_data.model_dump(exclude={"id"})
        updated_data["owui_id"] = id
        updated_data["user_id"] = tool_data.get("user_id")
        updated_data["specs"] = specs

        updated_record = await client.update(Collections.TOOLS, record.id, updated_data)

        # Convert back to ToolModel
        result_data = updated_record.model_dump()
        result_data["id"] = result_data.get("owui_id", result_data.get("id"))
        return ToolModel(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# DeleteToolsById
############################


@router.delete("/id/{id}/delete", response_model=bool)
async def delete_tools_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        # Find the tool by ID
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        tool_data = record.model_dump()

        # Check access
        if (
            tool_data.get("user_id") != user.id
            and not has_access(user.id, "write", tool_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        # Delete from BlueNexus
        await client.delete(Collections.TOOLS, record.id)

        # Clean up from app state
        TOOLS = request.app.state.TOOLS
        if id in TOOLS:
            del TOOLS[id]

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetToolValves
############################


@router.get("/id/{id}/valves", response_model=Optional[dict])
async def get_tools_valves_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        tool_data = response.get_records()[0].model_dump()
        return tool_data.get("valves", {})

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetToolValvesSpec
############################


@router.get("/id/{id}/valves/spec", response_model=Optional[dict])
async def get_tools_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id, user_id=user.id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "Valves"):
            Valves = tools_module.Valves
            return Valves.schema()
        return None

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateToolValves
############################


@router.post("/id/{id}/valves/update", response_model=Optional[dict])
async def update_tools_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        tool_data = record.model_dump()

        # Check access
        if (
            tool_data.get("user_id") != user.id
            and not has_access(user.id, "write", tool_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id, user_id=user.id)
            request.app.state.TOOLS[id] = tools_module

        if not hasattr(tools_module, "Valves"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
        Valves = tools_module.Valves

        # Validate and update valves
        form_data = {k: v for k, v in form_data.items() if v is not None}
        valves = Valves(**form_data)
        valves_dict = valves.model_dump(exclude_unset=True)

        # Update tool with new valves in BlueNexus
        tool_data["valves"] = valves_dict
        await client.update(Collections.TOOLS, record.id, tool_data)

        return valves_dict

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Failed to update tool valves by id {id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# ToolUserValves
############################


@router.get("/id/{id}/valves/user", response_model=Optional[dict])
async def get_tools_user_valves_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        tool_data = response.get_records()[0].model_dump()
        user_valves_all = tool_data.get("user_valves", {})
        return user_valves_all.get(user.id, {})

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


@router.get("/id/{id}/valves/user/spec", response_model=Optional[dict])
async def get_tools_user_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id, user_id=user.id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "UserValves"):
            UserValves = tools_module.UserValves
            return UserValves.schema()
        return None

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


@router.post("/id/{id}/valves/user/update", response_model=Optional[dict])
async def update_tools_user_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        tool_data = record.model_dump()

        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id, user_id=user.id)
            request.app.state.TOOLS[id] = tools_module

        if not hasattr(tools_module, "UserValves"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        UserValves = tools_module.UserValves

        # Validate and update user valves
        form_data = {k: v for k, v in form_data.items() if v is not None}
        user_valves = UserValves(**form_data)
        user_valves_dict = user_valves.model_dump(exclude_unset=True)

        # Update tool with user valves in BlueNexus
        user_valves_all = tool_data.get("user_valves", {})
        user_valves_all[user.id] = user_valves_dict
        tool_data["user_valves"] = user_valves_all
        await client.update(Collections.TOOLS, record.id, tool_data)

        return user_valves_dict

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Failed to update user valves by id {id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )
