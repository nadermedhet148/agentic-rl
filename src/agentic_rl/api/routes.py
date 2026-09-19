from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agentic_rl.core.agent import Agent
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
    return await _agent(request).run(body.message, source="user")


@router.post("/confirm/{episode_id}", response_model=Episode)
async def confirm(episode_id: str, request: Request) -> Episode:
    try:
        return await _agent(request).confirm(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/feedback", response_model=Episode)
async def feedback(body: Feedback, request: Request) -> Episode:
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
