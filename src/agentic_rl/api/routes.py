from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentic_rl.core import observability
from agentic_rl.core.agent import Agent, EventCallback
from agentic_rl.core.memory import MemoryStore
from agentic_rl.core.models import Episode, Feedback, Memory
from agentic_rl.core.store import EpisodeStore
from agentic_rl.scheduler.scheduler import AgentScheduler

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class AddMemoryRequest(BaseModel):
    text: str
    capability: str | None = None


def _agent(request: Request) -> Agent:
    return request.app.state.agent


def _store(request: Request) -> EpisodeStore:
    return request.app.state.store


def _scheduler(request: Request) -> AgentScheduler:
    return request.app.state.scheduler


def _memory(request: Request) -> MemoryStore:
    return request.app.state.memory


@router.post("/chat", response_model=Episode)
async def chat(body: ChatRequest, request: Request) -> Episode:
    episode_id = uuid4().hex
    with observability.trace("api.chat", input=body.message, trace_id=episode_id, session_id=episode_id):
        return await _agent(request).run(body.message, source="user", episode_id=episode_id)


def _sse(event: str, data: BaseModel | dict) -> str:
    body = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {body}\n\n"


def _stream(request: Request, run: Callable[[EventCallback], Awaitable[Episode]]) -> StreamingResponse:
    """Runs `run` (an agent.run/confirm call) in a detached background task so a
    client disconnect never aborts a run mid-episode, relaying its on_event
    callbacks plus a final done/error event through an asyncio.Queue as SSE."""
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    async def worker() -> None:
        try:
            episode = await run(lambda name, data: queue.put_nowait((name, data)))
            queue.put_nowait(("done", episode))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error event
            queue.put_nowait(("error", {"detail": str(exc)}))
        finally:
            queue.put_nowait(None)

    tasks: set = request.app.state.background_tasks
    task = asyncio.create_task(worker())
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    async def gen():
        while (item := await queue.get()) is not None:
            yield _sse(*item)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    episode_id = uuid4().hex

    async def run(on_event: EventCallback) -> Episode:
        with observability.trace("api.chat", input=body.message, trace_id=episode_id, session_id=episode_id):
            return await _agent(request).run(
                body.message, source="user", episode_id=episode_id, on_event=on_event
            )

    return _stream(request, run)


@router.post("/confirm/{episode_id}/stream")
async def confirm_stream(episode_id: str, request: Request) -> StreamingResponse:
    episode = _store(request).get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"unknown episode: {episode_id}")
    if episode.status != "pending_confirmation":
        raise HTTPException(
            status_code=409,
            detail=f"episode {episode_id} is not pending confirmation (status={episode.status})",
        )

    async def run(on_event: EventCallback) -> Episode:
        with observability.trace("api.confirm", input=episode_id, trace_id=episode_id, session_id=episode_id):
            return await _agent(request).confirm(episode_id, on_event=on_event)

    return _stream(request, run)


@router.post("/confirm/{episode_id}", response_model=Episode)
async def confirm(episode_id: str, request: Request) -> Episode:
    with observability.trace("api.confirm", input=episode_id, trace_id=episode_id, session_id=episode_id):
        try:
            return await _agent(request).confirm(episode_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/feedback", response_model=Episode)
async def feedback(body: Feedback, request: Request) -> Episode:
    with observability.trace("api.feedback", input=body.episode_id, trace_id=body.episode_id, session_id=body.episode_id):
        try:
            return await _agent(request).record_feedback(body)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/episodes", response_model=list[Episode])
def list_episodes(request: Request, limit: int = 50) -> list[Episode]:
    return _store(request).list_episodes(limit=limit)


@router.get("/metrics")
def metrics(request: Request, n: int = 100) -> dict[str, Any]:
    rewards = _store(request).rolling_reward(n=n)
    mean = sum(rewards) / len(rewards) if rewards else None
    return {"n": len(rewards), "rewards": rewards, "mean_reward": mean}


@router.get("/tasks")
def list_tasks(request: Request) -> list[dict]:
    return _scheduler(request).list_jobs()


@router.delete("/tasks/{job_id}")
def cancel_task(job_id: str, request: Request) -> dict:
    try:
        _scheduler(request).remove_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}") from exc
    _agent(request).cancel_task(job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.get("/memories", response_model=list[Memory])
def list_memories(request: Request, limit: int = 50) -> list[Memory]:
    return _memory(request).active_rules(limit=limit)


@router.post("/memories", response_model=Memory)
def add_memory(body: AddMemoryRequest, request: Request) -> Memory:
    return _memory(request).add(body.text, capability=body.capability)


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, request: Request) -> dict:
    result = _memory(request).deactivate(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown memory: {memory_id}")
    return {"status": "deactivated", "id": memory_id}
