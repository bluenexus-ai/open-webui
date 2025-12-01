import asyncio
import json
import logging
import random
import requests
import aiohttp
import urllib.parse
import urllib.request
from typing import Optional

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from open_webui.env import SRC_LOG_LEVELS
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["COMFYUI"])

default_headers = {"User-Agent": "Mozilla/5.0"}


def queue_prompt(prompt, client_id, base_url, api_key):
    log.info("queue_prompt")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    log.debug(f"queue_prompt data: {data}")
    try:
        req = urllib.request.Request(
            f"{base_url}/prompt",
            data=data,
            headers={**default_headers, "Authorization": f"Bearer {api_key}"},
        )
        response = urllib.request.urlopen(req).read()
        return json.loads(response)
    except Exception as e:
        log.exception(f"Error while queuing prompt: {e}")
        raise e


def get_image(filename, subfolder, folder_type, base_url, api_key):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    req = urllib.request.Request(
        f"{base_url}/view?{url_values}",
        headers={**default_headers, "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as response:
        return response.read()


def get_image_url(filename, subfolder, folder_type, base_url):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    return f"{base_url}/view?{url_values}"


def get_history(prompt_id, base_url, api_key):
    log.info("get_history")

    req = urllib.request.Request(
        f"{base_url}/history/{prompt_id}",
        headers={**default_headers, "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def get_images(ws, prompt, client_id, base_url, api_key):
    prompt_id = queue_prompt(prompt, client_id, base_url, api_key)["prompt_id"]
    log.info(f"Queued prompt with ID: {prompt_id}")
    output_images = []
    execution_error = None
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            log.debug(f"WebSocket message: {message.get('type')}")
            if message["type"] == "executing":
                data = message["data"]
                log.info(f"Executing node: {data.get('node')} for prompt: {data.get('prompt_id')}")
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    log.info("Workflow execution completed")
                    break  # Execution is done
            elif message["type"] == "execution_error":
                error_data = message.get("data", {})
                error_msg = error_data.get("exception_message", "Unknown error")
                node_type = error_data.get("node_type", "Unknown node")
                log.error(f"Workflow execution error in {node_type}: {error_msg}")
                execution_error = f"ComfyUI workflow error in {node_type}: {error_msg}"
        else:
            continue  # previews are binary data

    # Raise error if workflow failed
    if execution_error:
        raise Exception(execution_error)

    history_response = get_history(prompt_id, base_url, api_key)
    log.info(f"History response keys: {history_response.keys() if history_response else 'None'}")

    if prompt_id not in history_response:
        log.error(f"Prompt ID {prompt_id} not found in history response")
        return {"data": []}

    history = history_response[prompt_id]
    log.info(f"History keys: {history.keys() if history else 'None'}")
    log.info(f"History outputs: {history.get('outputs', {})}")

    outputs = history.get("outputs", {})
    for node_id in outputs:
        node_output = outputs[node_id]
        log.info(f"Node {node_id} output keys: {node_output.keys() if node_output else 'None'}")
        if "images" in node_output:
            for image in node_output["images"]:
                log.info(f"Found image: {image}")
                url = get_image_url(
                    image["filename"], image["subfolder"], image["type"], base_url
                )
                output_images.append({"url": url})

    log.info(f"Total output images found: {len(output_images)}")
    return {"data": output_images}


async def comfyui_upload_image(image_file_item, base_url, api_key):
    # ComfyUI endpoint is /upload/image (not /api/upload/image)
    base_url = base_url.strip().rstrip("/")
    url = f"{base_url}/upload/image"
    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _, (filename, file_bytes, mime_type) = image_file_item

    log.info(f"Uploading image to ComfyUI: {filename}, mime_type: {mime_type}, url: {url}")

    # Read bytes from BytesIO if needed
    if hasattr(file_bytes, 'read'):
        file_bytes.seek(0)
        file_data = file_bytes.read()
    else:
        file_data = file_bytes

    log.info(f"Image data size: {len(file_data)} bytes")

    form = aiohttp.FormData()
    form.add_field("image", file_data, filename=filename, content_type=mime_type)
    form.add_field("type", "input")  # required by ComfyUI
    form.add_field("overwrite", "true")  # overwrite if exists

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, headers=headers) as resp:
            log.info(f"ComfyUI upload response status: {resp.status}")
            if resp.status != 200:
                error_text = await resp.text()
                log.error(f"ComfyUI upload error: {error_text}")
                raise Exception(f"ComfyUI upload failed: {resp.status} - {error_text}")
            result = await resp.json()
            log.info(f"ComfyUI upload result: {result}")
            return result


class ComfyUINodeInput(BaseModel):
    type: Optional[str] = None
    node_ids: list[str] = []
    key: Optional[str] = "text"
    value: Optional[str] = None


class ComfyUIWorkflow(BaseModel):
    workflow: str
    nodes: list[ComfyUINodeInput]


class ComfyUICreateImageForm(BaseModel):
    workflow: ComfyUIWorkflow

    prompt: str
    negative_prompt: Optional[str] = None
    width: int
    height: int
    n: int = 1

    steps: Optional[int] = None
    seed: Optional[int] = None


async def comfyui_create_image(
    model: str, payload: ComfyUICreateImageForm, client_id, base_url, api_key
):
    base_url = base_url.strip()
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    workflow = json.loads(payload.workflow.workflow)

    for node in payload.workflow.nodes:
        if node.type:
            if node.type == "model":
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][node.key] = model
            elif node.type == "prompt":
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][
                        node.key if node.key else "text"
                    ] = payload.prompt
            elif node.type == "negative_prompt":
                # Use empty string if negative_prompt is None
                neg_prompt = payload.negative_prompt if payload.negative_prompt is not None else ""
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][
                        node.key if node.key else "text"
                    ] = neg_prompt
            elif node.type == "width":
                if payload.width is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "width"
                        ] = payload.width
            elif node.type == "height":
                if payload.height is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "height"
                        ] = payload.height
            elif node.type == "n":
                if payload.n is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "batch_size"
                        ] = payload.n
            elif node.type == "steps":
                if payload.steps is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "steps"
                        ] = payload.steps
            elif node.type == "seed":
                seed = (
                    payload.seed
                    if payload.seed
                    else random.randint(0, 1125899906842624)
                )
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][node.key] = seed
        else:
            for node_id in node.node_ids:
                workflow[node_id]["inputs"][node.key] = node.value

    try:
        ws = websocket.WebSocket()
        headers = {"Authorization": f"Bearer {api_key}"}
        ws.connect(f"{ws_url}/ws?clientId={client_id}", header=headers)
        log.info("WebSocket connection established.")
    except Exception as e:
        log.exception(f"Failed to connect to WebSocket server: {e}")
        return None

    try:
        log.info("Sending workflow to WebSocket server.")
        log.info(f"Workflow: {workflow}")
        images = await asyncio.to_thread(
            get_images, ws, workflow, client_id, base_url, api_key
        )
    except Exception as e:
        log.exception(f"Error while receiving images: {e}")
        images = None

    ws.close()

    return images


class ComfyUIEditImageForm(BaseModel):
    workflow: ComfyUIWorkflow

    image: str | list[str]
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    n: Optional[int] = None

    steps: Optional[int] = None
    seed: Optional[int] = None


async def comfyui_edit_image(
    model: str, payload: ComfyUIEditImageForm, client_id, base_url, api_key
):
    base_url = base_url.strip()
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    workflow = json.loads(payload.workflow.workflow)

    log.info(f"comfyui_edit_image called with model: {model}")
    log.info(f"Image(s) to edit: {payload.image}")
    log.info(f"Number of workflow nodes: {len(payload.workflow.nodes)}")
    log.info(f"Workflow nodes: {[{'type': n.type, 'node_ids': n.node_ids, 'key': n.key} for n in payload.workflow.nodes]}")

    for node in payload.workflow.nodes:
        if node.type:
            if node.type == "model":
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][node.key] = model
            elif node.type == "image":
                if isinstance(payload.image, list):
                    # check if multiple images are provided
                    for idx, node_id in enumerate(node.node_ids):
                        if idx < len(payload.image):
                            workflow[node_id]["inputs"][node.key] = payload.image[idx]
                else:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][node.key] = payload.image
            elif node.type == "prompt":
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][
                        node.key if node.key else "text"
                    ] = payload.prompt
            elif node.type == "negative_prompt":
                # Use empty string if negative_prompt is None
                neg_prompt = payload.negative_prompt if payload.negative_prompt is not None else ""
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][
                        node.key if node.key else "text"
                    ] = neg_prompt
            elif node.type == "width":
                if payload.width is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "width"
                        ] = payload.width
            elif node.type == "height":
                if payload.height is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "height"
                        ] = payload.height
            elif node.type == "n":
                if payload.n is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "batch_size"
                        ] = payload.n
            elif node.type == "steps":
                if payload.steps is not None:
                    for node_id in node.node_ids:
                        workflow[node_id]["inputs"][
                            node.key if node.key else "steps"
                        ] = payload.steps
            elif node.type == "seed":
                seed = (
                    payload.seed
                    if payload.seed
                    else random.randint(0, 1125899906842624)
                )
                for node_id in node.node_ids:
                    workflow[node_id]["inputs"][node.key] = seed
        else:
            for node_id in node.node_ids:
                workflow[node_id]["inputs"][node.key] = node.value

    try:
        ws = websocket.WebSocket()
        headers = {"Authorization": f"Bearer {api_key}"}
        ws.connect(f"{ws_url}/ws?clientId={client_id}", header=headers)
        log.info("WebSocket connection established.")
    except Exception as e:
        log.exception(f"Failed to connect to WebSocket server: {e}")
        return None

    try:
        log.info("Sending workflow to WebSocket server.")
        log.info(f"Workflow: {workflow}")
        images = await asyncio.to_thread(
            get_images, ws, workflow, client_id, base_url, api_key
        )
    except Exception as e:
        log.exception(f"Error while receiving images: {e}")
        images = None

    ws.close()

    return images
