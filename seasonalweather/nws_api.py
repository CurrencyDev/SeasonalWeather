from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


DEFAULT_UA = "SeasonalWeather/2.0 (automated IP radio system for weather; contact: info@seasonalnet.org)"


@dataclass
class NWSProduct:
    product_id: str
    product_text: str
    issuance_time: str | None = None
    product_type: str | None = None
    wfo: str | None = None


class NWSApi:
    def __init__(self, timeout: float = 8.0, user_agent: str = DEFAULT_UA) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/geo+json, application/ld+json, application/json",
            },
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # simple retry loop
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                r = await self._client.get(url, params=params)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"NWS API request failed: {url}") from last_exc

    async def latest_product_id(self, product_type: str, wfo: str) -> Optional[str]:
        url = f"https://api.weather.gov/products/types/{product_type}/locations/{wfo}"
        data = await self._get_json(url)

        items: List[Dict[str, Any]] = []
        if isinstance(data.get("products"), list):
            items = data["products"]
        elif isinstance(data.get("@graph"), list):
            items = data["@graph"]
        elif isinstance(data.get("graph"), list):
            items = data["graph"]

        for item in items:
            pid = item.get("id") or item.get("@id") or item.get("productId")
            if isinstance(pid, str) and pid:
                # pid might be full URL
                return pid.rstrip("/").split("/")[-1]
        return None

    async def get_product(self, product_id: str) -> Optional[NWSProduct]:
        url = f"https://api.weather.gov/products/{product_id}"
        data = await self._get_json(url)
        text = data.get("productText") or data.get("product_text") or data.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return NWSProduct(
            product_id=product_id,
            product_text=text,
            issuance_time=data.get("issuanceTime") or data.get("issuance_time"),
            product_type=data.get("productCode") or data.get("product_code"),
            wfo=data.get("wfo") or data.get("wfoCode"),
        )

    async def point_forecast_periods(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        point = await self._get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        props = point.get("properties", {})
        forecast_url = props.get("forecast")
        if not isinstance(forecast_url, str) or not forecast_url:
            return []
        fc = await self._get_json(forecast_url)
        return list((fc.get("properties", {}) or {}).get("periods", []) or [])

    async def latest_observation(self, station_id: str) -> Optional[Dict[str, Any]]:
        data = await self._get_json(f"https://api.weather.gov/stations/{station_id}/observations/latest")
        return data.get("properties")

    async def active_alerts(self, areas: List[str]) -> List[Dict[str, Any]]:
        # areas = state/territory abbreviations (e.g. ["MD","VA","DC","WV"])
        params = {"area": ",".join(areas), "status": "actual"}
        data = await self._get_json("https://api.weather.gov/alerts/active", params=params)
        feats = data.get("features")
        return list(feats) if isinstance(feats, list) else []
