from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


API_VERSION = "1.1.0"
PROBLEM_JSON = "application/problem+json"

PROBLEM_DETAILS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["type", "title", "status"],
    "properties": {
        "type": {
            "type": "string",
            "format": "uri-reference",
            "description": "Problem type URI or URI-reference.",
        },
        "title": {"type": "string"},
        "status": {"type": "integer", "minimum": 100, "maximum": 599},
        "detail": {"type": "string"},
        "instance": {"type": "string", "format": "uri-reference"},
        "code": {
            "type": "string",
            "description": "Stable SeasonalWeather error code extension.",
        },
        "details": {
            "type": "object",
            "additionalProperties": True,
            "description": "Machine-readable error details extension.",
        },
        "errors": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
            "description": "Validation-error detail list extension.",
        },
        "request_id": {
            "type": "string",
            "description": "Request correlation identifier also returned as X-Request-ID.",
        },
    },
}

HEALTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["ok", "liquidsoap_telnet", "api"],
    "properties": {
        "ok": {"type": "boolean"},
        "liquidsoap_telnet": {
            "type": "object",
            "additionalProperties": True,
            "properties": {"reachable": {"type": "boolean"}},
        },
        "api": {
            "type": "object",
            "additionalProperties": True,
            "properties": {"version": {"type": "string"}},
        },
    },
}

STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": [
        "mode",
        "heightened_until",
        "last_product_desc",
        "liquidsoap_telnet_reachable",
        "nwws_queue_size",
        "cap_queue_size",
        "ern_queue_size",
        "config_sha256",
    ],
    "properties": {
        "mode": {"type": "string"},
        "heightened_until": {"type": ["string", "null"], "format": "date-time"},
        "last_heightened_at": {"type": ["string", "null"], "format": "date-time"},
        "last_product_desc": {"type": ["string", "null"]},
        "liquidsoap_telnet_reachable": {"type": "boolean"},
        "nwws_queue_size": {"type": "integer", "minimum": 0},
        "cap_queue_size": {"type": "integer", "minimum": 0},
        "ern_queue_size": {"type": "integer", "minimum": 0},
        "config_sha256": {"type": ["string", "null"]},
    },
}

STATION_FEED_ALERT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["id", "event", "headline"],
    "properties": {
        "id": {"type": "string"},
        "event": {"type": "string"},
        "headline": {"type": "string"},
        "severity": {"type": "string"},
        "urgency": {"type": "string"},
        "certainty": {"type": "string"},
        "area": {"type": "string"},
        "effective": {"type": ["string", "null"], "format": "date-time"},
        "ends": {"type": ["string", "null"], "format": "date-time"},
        "expires": {"type": ["string", "null"], "format": "date-time"},
        "sent": {"type": ["string", "null"], "format": "date-time"},
        "sameCodes": {"type": "array", "items": {"type": "string"}},
        "source": {"type": ["string", "null"]},
        "from": {
            "type": ["object", "null"],
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string", "enum": ["relay", "origin", "unknown"]},
            },
        },
        "links": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "primary": {"type": "string"},
                "nws": {"type": "string"},
            },
        },
    },
}

STATION_FEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["stationId", "generatedAt", "source", "alerts"],
    "properties": {
        "stationId": {"type": "string"},
        "generatedAt": {"type": "string", "format": "date-time"},
        "source": {"type": "string"},
        "alerts": {
            "type": "array",
            "items": {"$ref": "#/components/schemas/StationFeedAlert"},
        },
    },
}

CONFIG_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["config_path", "station", "cycle", "observations", "nwws", "policy", "tts"],
    "properties": {
        "config_path": {"type": "string"},
        "config_sha256": {"type": ["string", "null"]},
        "station": {"type": "object", "additionalProperties": True},
        "cycle": {"type": "object", "additionalProperties": True},
        "observations": {"type": "object", "additionalProperties": True},
        "nwws": {"type": "object", "additionalProperties": True},
        "policy": {"type": "object", "additionalProperties": True},
        "tts": {"type": "object", "additionalProperties": True},
    },
}


def json_response(description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": schema,
            },
        },
    }


def problem_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            PROBLEM_JSON: {
                "schema": {"$ref": "#/components/schemas/ProblemDetails"},
            },
        },
    }


STANDARD_PROBLEM_RESPONSES: dict[int, dict[str, Any]] = {
    400: problem_response("Bad request."),
    401: problem_response("Authentication is required."),
    403: problem_response("The authenticated principal lacks access to this resource."),
    404: problem_response("The requested resource was not found."),
    409: problem_response("The request conflicts with existing state."),
    422: problem_response("The request body, query, path, or headers failed validation."),
    500: problem_response("The server failed while handling the request."),
    503: problem_response("A required backend dependency is unavailable."),
}

PUBLIC_PROBLEM_RESPONSES: dict[int, dict[str, Any]] = {
    422: STANDARD_PROBLEM_RESPONSES[422],
    500: STANDARD_PROBLEM_RESPONSES[500],
}


def install_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title="SeasonalWeather API",
            version=API_VERSION,
            summary="Weather radio automation and control API.",
            description=(
                "SeasonalWeather exposes public station-feed data plus authenticated "
                "operator/control-plane endpoints. Error responses use RFC 9457 "
                "Problem Details with the application/problem+json media type."
            ),
            routes=app.routes,
            openapi_version="3.1.0",
            tags=[
                {"name": "status", "description": "Health and runtime status."},
                {"name": "station-feed", "description": "Station handled-alert feeds."},
                {"name": "commands", "description": "Command status and event streams."},
                {"name": "control", "description": "Authenticated station control operations."},
                {"name": "origination", "description": "Authenticated test and manual alert origination."},
                {"name": "configuration", "description": "Authenticated configuration inspection and reload."},
            ],
        )
        schema["jsonSchemaDialect"] = "https://json-schema.org/draft/2020-12/schema"
        components = schema.setdefault("components", {})
        schemas = components.setdefault("schemas", {})
        schemas.update(
            {
                "ProblemDetails": PROBLEM_DETAILS_SCHEMA,
                "Health": HEALTH_SCHEMA,
                "RuntimeStatus": STATUS_SCHEMA,
                "StationFeedAlert": STATION_FEED_ALERT_SCHEMA,
                "StationFeed": STATION_FEED_SCHEMA,
                "ConfigSummary": CONFIG_SUMMARY_SCHEMA,
            }
        )
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
