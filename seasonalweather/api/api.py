from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    AudioUploadAccepted,
    ClearHeightenedModeRequest,
    CommandAccepted,
    CommandSnapshot,
    ConfigReloadRequest,
    ErrorEnvelope,
    OriginateAudioRequest,
    OriginateTestRequest,
    OriginateTextRequest,
    RebuildCycleRequest,
    SetHeightenedModeRequest,
)
from .auth import ApiPrincipal, get_api_principal, require_scopes
from .commands import CommandNotFoundError, CommandStore, IdempotencyConflictError
from ..control import ControlError, OrchestratorControl


def _request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def _error_response(*, status_code: int, request_id: str, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    payload = ErrorEnvelope(error={"code": code, "message": message, "details": details or {}}, request_id=request_id)
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _command_accepted(record: Any, *, replayed: bool) -> CommandAccepted:
    return CommandAccepted(
        command_id=record.command_id,
        command_type=record.command_type,
        status=record.status,
        accepted_at=record.accepted_at,
        idempotent_replay=replayed,
        request_id=record.request_id,
    )


def _command_snapshot(record: Any) -> CommandSnapshot:
    return CommandSnapshot.model_validate(record.snapshot())


async def _require_idempotency_key(idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> str:
    key = (idempotency_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail={"code": "missing_idempotency_key", "message": "Idempotency-Key header is required."})
    if len(key) > 200:
        raise HTTPException(status_code=400, detail={"code": "invalid_idempotency_key", "message": "Idempotency-Key is too long."})
    return key


async def _execute_command(
    *,
    store: CommandStore,
    principal: ApiPrincipal,
    idempotency_key: str,
    command_type: str,
    payload: dict[str, Any],
    action: Callable[[], Awaitable[dict[str, Any]]],
    success_event: str | None = None,
) -> CommandAccepted:
    try:
        record, replayed = await store.create_or_replay(
            command_type=command_type,
            idempotency_key=idempotency_key,
            actor=principal.subject,
            payload=payload,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail={"code": "idempotency_conflict", "message": str(exc)}) from exc

    if replayed:
        return _command_accepted(record, replayed=True)

    await store.mark_running(record.command_id)
    try:
        result = await action()
    except ControlError as exc:
        await store.mark_failed(record.command_id, exc.to_dict())
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
    except Exception as exc:
        err = {"code": "internal_error", "message": "Unhandled server error while executing command."}
        await store.mark_failed(record.command_id, err)
        raise HTTPException(status_code=500, detail=err) from exc

    await store.mark_succeeded(record.command_id, result)
    if success_event:
        await store.broker.publish(success_event, {"command_id": record.command_id, "command_type": record.command_type, "result": result})
    return _command_accepted(record, replayed=False)


def create_app(control: OrchestratorControl, *, store: CommandStore | None = None) -> FastAPI:
    command_store = store or CommandStore()
    app = FastAPI(title="SeasonalWeather Control API", version="1.0")
    app.state.control = control
    app.state.command_store = command_store

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status_code=422,
            request_id=_request_id(),
            code="request_validation_failed",
            message="Request body, path, query, or header validation failed.",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        return _error_response(
            status_code=exc.status_code,
            request_id=_request_id(),
            code=str(detail.get("code") or "http_error"),
            message=str(detail.get("message") or "HTTP error"),
            details=detail.get("details") if isinstance(detail.get("details"), dict) else {},
        )

    @app.get("/healthz")
    async def healthz(principal: ApiPrincipal = Depends(require_scopes("read:health"))) -> dict[str, Any]:
        return await control.get_health()

    @app.get("/v1/health")
    async def v1_health(principal: ApiPrincipal = Depends(require_scopes("read:health"))) -> dict[str, Any]:
        return await control.get_health()

    @app.get("/v1/status")
    async def v1_status(principal: ApiPrincipal = Depends(require_scopes("read:status"))) -> dict[str, Any]:
        return await control.get_status()

    @app.get("/v1/station-feed")
    async def v1_station_feed(principal: ApiPrincipal = Depends(require_scopes("read:alerts"))) -> dict[str, Any]:
        return await control.get_station_feed()

    @app.get("/v1/config/summary")
    async def v1_config_summary(principal: ApiPrincipal = Depends(require_scopes("read:config"))) -> dict[str, Any]:
        return await control.get_config_summary()

    @app.get("/v1/commands/{command_id}", response_model=CommandSnapshot)
    async def v1_command(command_id: str, principal: ApiPrincipal = Depends(require_scopes("read:status"))) -> CommandSnapshot:
        try:
            record = await command_store.get(command_id)
        except CommandNotFoundError as exc:
            raise HTTPException(status_code=404, detail={"code": "command_not_found", "message": "Command was not found."}) from exc
        return _command_snapshot(record)

    @app.post("/v1/cycle/rebuild", response_model=CommandAccepted)
    async def v1_cycle_rebuild(
        req: RebuildCycleRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:cycle")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="cycle.rebuild",
            payload=payload,
            action=lambda: control.rebuild_cycle(reason=req.reason, actor=principal.subject),
            success_event="cycle.rebuild.completed",
        )

    @app.post("/v1/mode/heightened", response_model=CommandAccepted)
    async def v1_mode_heightened(
        req: SetHeightenedModeRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:mode")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="mode.heightened.set",
            payload=payload,
            action=lambda: control.set_heightened_mode(minutes=req.minutes, reason=req.reason, actor=principal.subject),
            success_event="mode.changed",
        )

    @app.delete("/v1/mode/heightened", response_model=CommandAccepted)
    async def v1_mode_heightened_clear(
        req: ClearHeightenedModeRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:mode")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="mode.heightened.clear",
            payload=payload,
            action=lambda: control.clear_heightened_mode(reason=req.reason, actor=principal.subject),
            success_event="mode.changed",
        )

    @app.post("/v1/tests/originate", response_model=CommandAccepted)
    async def v1_tests_originate(
        req: OriginateTestRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:tests")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="tests.originate",
            payload=payload,
            action=lambda: control.originate_test(event_code=req.event_code, actor=principal.subject),
            success_event="alert.originated",
        )

    @app.post("/v1/uploads/audio", response_model=AudioUploadAccepted)
    async def v1_upload_audio(
        file: UploadFile = File(...),
        principal: ApiPrincipal = Depends(require_scopes("control:audio")),
    ) -> AudioUploadAccepted:
        data = await file.read()
        try:
            payload = await control.stage_wav_upload(
                filename=file.filename or "upload.wav",
                content_type=file.content_type or "audio/wav",
                data=data,
                actor=principal.subject,
            )
        except ControlError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
        return AudioUploadAccepted.model_validate(payload)

    @app.post("/v1/originate/text", response_model=CommandAccepted)
    async def v1_originate_text(
        req: OriginateTextRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:originate")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="originate.text",
            payload=payload,
            action=lambda: control.originate_text(req, actor=principal.subject),
            success_event="alert.originated",
        )

    @app.post("/v1/originate/audio", response_model=CommandAccepted)
    async def v1_originate_audio(
        req: OriginateAudioRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:originate")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="originate.audio",
            payload=payload,
            action=lambda: control.originate_audio(req, actor=principal.subject),
            success_event="alert.originated",
        )

    @app.post("/v1/config/reload", response_model=CommandAccepted)
    async def v1_config_reload(
        req: ConfigReloadRequest,
        principal: ApiPrincipal = Depends(require_scopes("control:config")),
        idempotency_key: str = Depends(_require_idempotency_key),
    ) -> CommandAccepted:
        payload = req.model_dump(mode="json")
        return await _execute_command(
            store=command_store,
            principal=principal,
            idempotency_key=idempotency_key,
            command_type="config.reload",
            payload=payload,
            action=lambda: control.reload_config(actor=principal.subject, reason=req.reason),
            success_event="config.reloaded",
        )

    @app.get("/v1/events")
    async def v1_events(principal: ApiPrincipal = Depends(require_scopes("read:status"))) -> StreamingResponse:
        queue = await command_store.broker.subscribe()

        async def _stream() -> Any:
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    payload = json.dumps(item["data"], separators=(",", ":"), ensure_ascii=False)
                    yield f"event: {item['event']}\n"
                    yield f"data: {payload}\n\n"
            finally:
                await command_store.broker.unsubscribe(queue)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app
