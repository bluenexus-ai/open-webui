import logging
from pathlib import Path
from typing import Optional
import time
import re
import aiohttp
from open_webui.models.groups import Groups
from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, Depends, HTTPException, Request, status


from open_webui.models.tools import (
    ToolForm,
    ToolModel,
    ToolResponse,
    ToolUserResponse,
    Tools,
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
from open_webui.utils.bluenexus.config import is_bluenexus_data_storage_enabled
from open_webui.repositories import get_tool_repository

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import CACHE_DIR, BYPASS_ADMIN_ACCESS_CONTROL
from open_webui.constants import ERROR_MESSAGES


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


router = APIRouter()


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

    # Get user's group IDs for access control check
    user_group_ids = []
    groups = Groups.get_groups_by_member_id(user.id) if hasattr(Groups, 'get_groups_by_member_id') else []
    user_group_ids = [g.id for g in groups] if groups else []

    # Local Tools
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)

        # Get all tools and filter by access
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            tools_data = await repo.get_all()
        else:
            # Get all tools to check access control (user's own + shared)
            tools_data = await repo.get_all()

        for tool_data in tools_data:
            # Check access: owner or has read access
            if (
                user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL
                or tool_data.get("user_id") == user.id
                or has_access(user.id, "read", tool_data.get("access_control"), user_group_ids)
            ):
                tool_id = tool_data.get("id")
                tool_module = get_tool_module(request, tool_id, load_from_db=False, user_id=user.id)
                tools.append(
                    ToolUserResponse(
                        **{
                            **tool_data,
                            "has_user_valves": hasattr(tool_module, "UserValves") if tool_module else False,
                        }
                    )
                )
    else:
        # PostgreSQL path - get all tools and filter by access
        all_tools = Tools.get_tools()
        for tool in all_tools:
            if (
                user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL
                or tool.user_id == user.id
                or has_access(user.id, "read", tool.access_control, user_group_ids)
            ):
                tool_module = get_tool_module(request, tool.id, load_from_db=False, user_id=user.id)
                tool_data = tool.model_dump()
                tool_data["has_user_valves"] = hasattr(tool_module, "UserValves") if tool_module else False
                tools.append(ToolUserResponse(**tool_data))

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
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)

        # Admins with bypass get all tools, others get their own
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            tools_data = await repo.get_all()
        else:
            tools_data = await repo.get_list(user.id)

        tools = []
        for tool_data in tools_data:
            # Apply access control for write permission
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                tools.append(ToolUserResponse(**tool_data))
            elif tool_data.get("user_id") == user.id or has_access(user.id, "write", tool_data.get("access_control")):
                tools.append(ToolUserResponse(**tool_data))

        return tools
    else:
        # PostgreSQL path
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            tools = Tools.get_tools()
        else:
            tools = Tools.get_tools_by_user_id(user.id)

        result = []
        for tool in tools:
            if has_access(user.id, "write", tool.access_control):
                result.append(ToolUserResponse(**tool.model_dump()))

        return result


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
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tools_data = await repo.get_all()

        tools = []
        for tool_data in tools_data:
            tools.append(ToolModel(**tool_data))

        return tools
    else:
        # PostgreSQL path
        return Tools.get_tools()


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

    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)

        # Check if tool with this ID already exists
        existing = await repo.get_by_id(form_data.id)
        if existing:
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

        # Create tool
        tool_data = form_data.model_dump()
        tool_data["specs"] = specs

        result = await repo.create(user.id, tool_data)
        return ToolResponse(**result) if result else None
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(form_data.id)
        if tool:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ID_TAKEN,
            )

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

        tool = Tools.insert_new_tool(user.id, form_data, specs)
        if tool:
            return ToolResponse(**tool.model_dump())
        return None


############################
# GetToolsById
############################


@router.get("/id/{id}", response_model=Optional[ToolModel])
async def get_tools_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            user.role == "admin"
            or tool.user_id == user.id
            or has_access(user.id, "read", tool.access_control)
        ):
            return tool
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
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
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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

        # Update tool
        update_data = form_data.model_dump(exclude={"id"})
        update_data["specs"] = specs

        result = await repo.update(id, update_data)
        return ToolModel(**result) if result else None
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            tool.user_id != user.id
            and not has_access(user.id, "write", tool.access_control)
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = load_tool_module_by_id(id, content=form_data.content)
        form_data.meta.manifest = frontmatter

        TOOLS = request.app.state.TOOLS
        TOOLS[id] = tool_module

        specs = get_tool_specs(TOOLS[id])

        tool = Tools.update_tool_by_id(id, {**form_data.model_dump(exclude={"id"}), "specs": specs})
        return tool


############################
# DeleteToolsById
############################


@router.delete("/id/{id}/delete", response_model=bool)
async def delete_tools_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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

        # Delete tool
        result = await repo.delete(id)

        # Clean up from app state
        TOOLS = request.app.state.TOOLS
        if id in TOOLS:
            del TOOLS[id]

        return result
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            tool.user_id != user.id
            and not has_access(user.id, "write", tool.access_control)
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        result = Tools.delete_tool_by_id(id)

        TOOLS = request.app.state.TOOLS
        if id in TOOLS:
            del TOOLS[id]

        return result


############################
# GetToolValves
############################


@router.get("/id/{id}/valves", response_model=Optional[dict])
async def get_tools_valves_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        return tool_data.get("valves", {})
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        return tool.valves if tool.valves else {}


############################
# GetToolValvesSpec
############################


@router.get("/id/{id}/valves/spec", response_model=Optional[dict])
async def get_tools_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        tool = Tools.get_tool_by_id(id)
        if not tool:
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


############################
# UpdateToolValves
############################


@router.post("/id/{id}/valves/update", response_model=Optional[dict])
async def update_tools_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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

        # Update tool with new valves
        tool_data["valves"] = valves_dict
        await repo.update(id, {"valves": valves_dict})

        return valves_dict
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            tool.user_id != user.id
            and not has_access(user.id, "write", tool.access_control)
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

        form_data = {k: v for k, v in form_data.items() if v is not None}
        valves = Valves(**form_data)
        valves_dict = valves.model_dump(exclude_unset=True)

        Tools.update_tool_by_id(id, {"valves": valves_dict})

        return valves_dict


############################
# ToolUserValves
############################


@router.get("/id/{id}/valves/user", response_model=Optional[dict])
async def get_tools_user_valves_by_id(id: str, user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        user_valves_all = tool_data.get("user_valves", {})
        return user_valves_all.get(user.id, {})
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        user_valves = tool.user_valves if tool.user_valves else {}
        return user_valves.get(user.id, {})


@router.get("/id/{id}/valves/user/spec", response_model=Optional[dict])
async def get_tools_user_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        tool = Tools.get_tool_by_id(id)
        if not tool:
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


@router.post("/id/{id}/valves/user/update", response_model=Optional[dict])
async def update_tools_user_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    if is_bluenexus_data_storage_enabled():
        repo = get_tool_repository(user.id)
        tool_data = await repo.get_by_id(id)

        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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

        # Update tool with user valves
        user_valves_all = tool_data.get("user_valves", {})
        user_valves_all[user.id] = user_valves_dict
        await repo.update(id, {"user_valves": user_valves_all})

        return user_valves_dict
    else:
        # PostgreSQL path
        tool = Tools.get_tool_by_id(id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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

        form_data = {k: v for k, v in form_data.items() if v is not None}
        user_valves = UserValves(**form_data)
        user_valves_dict = user_valves.model_dump(exclude_unset=True)

        user_valves_all = tool.user_valves if tool.user_valves else {}
        user_valves_all[user.id] = user_valves_dict
        Tools.update_tool_by_id(id, {"user_valves": user_valves_all})

        return user_valves_dict
