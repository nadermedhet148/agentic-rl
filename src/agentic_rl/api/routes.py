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
from agentic_rl.core.agent import EventCallback
from agentic_rl.core.hub import KnowledgeHub
from agentic_rl.core.memory import MemoryStore
from agentic_rl.core.models import Episode, Feedback, Memory, Session
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.core.team import Team
from agentic_rl.scheduler.scheduler import AgentScheduler

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    agent_id: str | None = None  # None = let the team's Router pick (core/router.py)


class AddMemoryRequest(BaseModel):
    text: str
    capability: str | None = None


def _team(request: Request) -> Team:
    return request.app.state.team


def _hub(request: Request) -> KnowledgeHub:
    return request.app.state.hub


def _check_agent(request: Request, agent_id: str | None) -> None:
    if agent_id is not None:
        try:
            _team(request).get(agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


def _store(request: Request) -> EpisodeStore:
    return request.app.state.store


def _scheduler(request: Request) -> AgentScheduler:
    return request.app.state.scheduler


def _memory(request: Request) -> MemoryStore:
    return request.app.state.memory


def _sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


@router.post("/chat", response_model=Episode)
async def chat(body: ChatRequest, request: Request) -> Episode:
    _check_agent(request, body.agent_id)
    episode_id = uuid4().hex
    with observability.trace("api.chat", input=body.message, trace_id=episode_id, session_id=episode_id):
        return await _team(request).run(
            body.message, source="user", episode_id=episode_id, session_id=body.session_id, agent_id=body.agent_id
        )


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
    _check_agent(request, body.agent_id)
    episode_id = uuid4().hex

    async def run(on_event: EventCallback) -> Episode:
        with observability.trace("api.chat", input=body.message, trace_id=episode_id, session_id=episode_id):
            return await _team(request).run(
                body.message,
                source="user",
                episode_id=episode_id,
                on_event=on_event,
                session_id=body.session_id,
                agent_id=body.agent_id,
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
            return await _team(request).confirm(episode_id, on_event=on_event)

    return _stream(request, run)


@router.post("/confirm/{episode_id}", response_model=Episode)
async def confirm(episode_id: str, request: Request) -> Episode:
    with observability.trace("api.confirm", input=episode_id, trace_id=episode_id, session_id=episode_id):
        try:
            return await _team(request).confirm(episode_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/feedback", response_model=Episode)
async def feedback(body: Feedback, request: Request) -> Episode:
    with observability.trace("api.feedback", input=body.episode_id, trace_id=body.episode_id, session_id=body.episode_id):
        try:
            return await _team(request).record_feedback(body)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/episodes", response_model=list[Episode])
def list_episodes(request: Request, limit: int = 50, agent_id: str | None = None) -> list[Episode]:
    return _store(request).list_episodes(limit=limit, agent_id=agent_id)


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
    _team(request).cancel_task(job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.get("/memories", response_model=list[Memory])
def list_memories(request: Request, limit: int = 50, agent_id: str | None = None) -> list[Memory]:
    """Every active rule, or — with `agent_id` — only those that agent sees."""
    return _memory(request).active_rules(limit=limit, agent_id=agent_id)


@router.post("/memories", response_model=Memory)
def add_memory(body: AddMemoryRequest, request: Request) -> Memory:
    return _memory(request).add(body.text, capability=body.capability)


@router.post("/memories/{memory_id}/promote", response_model=Memory)
def promote_memory(memory_id: str, request: Request) -> Memory:
    """Share a private rule with the whole team — the only way a rule that loosens
    safety (skips a confirmation) ever becomes team-wide."""
    result = _memory(request).promote(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown memory: {memory_id}")
    return result


@router.post("/memories/{memory_id}/demote", response_model=Memory)
def demote_memory(memory_id: str, request: Request) -> Memory:
    result = _memory(request).demote(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown memory: {memory_id}")
    return result


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, request: Request) -> dict:
    result = _memory(request).deactivate(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown memory: {memory_id}")
    return {"status": "deactivated", "id": memory_id}


@router.post("/sessions", response_model=Session)
def start_session(request: Request) -> Session:
    return _sessions(request).start()


@router.post("/sessions/{session_id}/end", response_model=Session)
def end_session(session_id: str, request: Request) -> Session:
    result = _sessions(request).end(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return result


# --- team (docs/MULTI-AGENT-PLAN.md) ----------------------------------------------


@router.get("/agents")
def list_agents(request: Request, n: int = 100) -> list[dict[str, Any]]:
    store = _store(request)
    result = []
    for agent in _team(request).agents():
        rewards = store.rolling_reward(n=n, agent_id=agent.id)
        result.append(
            {
                **agent.profile.model_dump(),
                "available_capabilities": agent.registry.names(),
                "policy_id": agent.policy.id,
                "recent_episodes": len(rewards),
                "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            }
        )
    return result


@router.get("/agents/trust")
def agent_trust(request: Request) -> list[dict[str, Any]]:
    """Learned trust between every ordered pair of agents (core/hub.py)."""
    return _hub(request).trust_matrix()


@router.get("/agents/{agent_id}/metrics")
def agent_metrics(agent_id: str, request: Request, n: int = 100) -> dict[str, Any]:
    _check_agent(request, agent_id)
    rewards = _store(request).rolling_reward(n=n, agent_id=agent_id)
    mean = sum(rewards) / len(rewards) if rewards else None
    return {"agent_id": agent_id, "n": len(rewards), "rewards": rewards, "mean_reward": mean}
