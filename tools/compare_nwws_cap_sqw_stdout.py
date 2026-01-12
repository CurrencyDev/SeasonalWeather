#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Ensure repo root is importable: tools/ -> app/
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seasonalweather.product import parse_product_text  # type: ignore
from seasonalweather.alert_builder import build_spoken_alert  # type: ignore


def _ua() -> str:
    return (
        os.environ.get("SEASONAL_CAP_USER_AGENT")
        or os.environ.get("SEASONAL_NWS_USER_AGENT")
        or "SeasonalWeather compare tool (stdlib)"
    )


def fetch_json(url: str, *, ua: str, timeout: int = 20) -> dict:
    req = Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def clean_cap_text(s: str, *, limit: int = 900) -> str:
    s2 = (s or "").replace("\r", " ").replace("\n", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    s2 = s2.replace("...", ". ").replace("..", ".")
    if len(s2) > limit:
        s2 = s2[:limit].rstrip() + "..."
    return s2


def build_cap_full_script(props: dict) -> str:
    event = clean_cap_text(props.get("event", "") or "", limit=120)
    headline = clean_cap_text(props.get("headline", "") or "", limit=280)
    area = clean_cap_text(props.get("areaDesc", "") or "", limit=320)
    desc = clean_cap_text(props.get("description", "") or "", limit=1200)
    instr = clean_cap_text(props.get("instruction", "") or "", limit=700)

    lines: list[str] = []
    lines.append("The National Weather Service has issued the following message.")
    if event:
        lines.append(f"{event}.")
    if headline:
        lines.append(headline if headline.endswith((".", "!", "?")) else headline + ".")
    if area:
        lines.append(f"For the following areas: {area}.")
    if desc:
        lines.append(desc)
    if instr:
        lines.append("Instructions.")
        lines.append(instr)
    lines.append("End of message.")
    return "\n".join(ln.strip() for ln in lines if ln and ln.strip()).strip()


def find_markers(label: str, text: str) -> None:
    markers = [
        "HAZARD",
        "IMPACT",
        "SOURCE",
        "PRECAUTIONARY/PREPAREDNESS ACTIONS",
        "PRECAUTIONARY",
        "PREPAREDNESS",
        "* WHAT",
        "* WHERE",
        "* WHEN",
        "* IMPACTS",
    ]
    t = text or ""
    print(f"\n--- Marker scan: {label} (len={len(t)}) ---")
    up = t.upper()
    for m in markers:
        i = up.find(m)
        if i >= 0:
            print(f"  hit: {m} @ {i}")
    # also count how many times the classic section divider appears
    print(f"  '&&' count: {t.count('&&')}")
    print(f"  '$$' count: {t.count('$$')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-id", default="4e0f4648-90b8-4cad-a03b-70250d76e366",
                    help="NWS product UUID (from your logs).")
    ap.add_argument("--cap-area", default="DC,MD,PA,VA,WV",
                    help="CAP active area list (comma separated).")
    ap.add_argument("--cap-match-vtec", default="KCTP.SQ.W.0020",
                    help="Substring to find in CAP VTEC parameter (track id).")
    ap.add_argument("--cap-match-sent", default="2026-01-11T16:00:00-05:00",
                    help="Optional CAP sent timestamp to match.")
    ap.add_argument("--cap-match-event", default="Snow Squall Warning",
                    help="Optional CAP event name to match.")
    ap.add_argument("--max-product-chars", type=int, default=8000,
                    help="How much of the raw productText to print.")
    ap.add_argument("--no-cap", action="store_true",
                    help="Skip CAP fetch (NWWS only).")
    args = ap.parse_args()

    ua = _ua()

    # ---- Fetch product text ----
    prod_url = f"https://api.weather.gov/products/{args.product_id}"
    try:
        prod = fetch_json(prod_url, ua=ua)
    except (HTTPError, URLError) as e:
        print(f"[FAIL] Could not fetch product JSON: {prod_url}\n{e}")
        return 2
    except Exception as e:
        print(f"[FAIL] Could not fetch product JSON: {prod_url}\n{e}")
        return 2

    product_text = (
        prod.get("productText")
        or prod.get("product_text")
        or prod.get("productText", "")
        or ""
    )

    print("\n==================== NWWS / PRODUCT JSON ====================")
    print(f"product_url: {prod_url}")
    print(f"productText len: {len(product_text)}")
    print("\n==================== OFFICIAL PRODUCT TEXT (head) ====================")
    head = product_text[: max(0, int(args.max_product_chars))]
    print(head)
    if len(product_text) > len(head):
        print("\n... [truncated] ...")

    # ---- NWWS: parse + build spoken script ----
    parsed = parse_product_text(product_text)
    if not parsed:
        # Duck-typed fallback if parse_product_text doesn't like the API text.
        parsed = SimpleNamespace(
            product_type="SQW",
            awips_id="SQWCTP",
            wfo="KCTP",
            raw_text=product_text,
        )

    spoken = build_spoken_alert(parsed, product_text)
    nwws_script = getattr(spoken, "script", "")
    if not isinstance(nwws_script, str):
        nwws_script = str(nwws_script)

    print("\n==================== NWWS build_spoken_alert().script ====================")
    print(nwws_script)
    find_markers("NWWS spoken.script", nwws_script)

    # ---- CAP: fetch active + find matching alert ----
    if args.no_cap:
        print("\n[OK] --no-cap set; skipping CAP.")
        return 0

    cap_url = f"https://api.weather.gov/alerts/active?area={args.cap_area.replace(',', '%2C')}&status=actual"
    try:
        cap = fetch_json(cap_url, ua=ua)
    except Exception as e:
        print(f"\n[WARN] Could not fetch CAP active feed:\n  {cap_url}\n  {e}")
        return 0

    feats = cap.get("features") or []
    best = None
    best_reason = ""

    want_vtec = (args.cap_match_vtec or "").strip()
    want_sent = (args.cap_match_sent or "").strip()
    want_event = (args.cap_match_event or "").strip()

    def vtec_blob(props: dict) -> str:
        params = props.get("parameters") if isinstance(props.get("parameters"), dict) else {}
        v = params.get("VTEC")
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return " ".join(str(x) for x in v)
        return ""

    for f in feats:
        props = f.get("properties") if isinstance(f.get("properties"), dict) else {}
        ev = str(props.get("event") or "")
        sent = str(props.get("sent") or "")
        vb = vtec_blob(props)

        # normalize
        vb_norm = vb.replace(" ", "")
        score = 0
        reasons = []

        if want_vtec and want_vtec in vb_norm:
            score += 5
            reasons.append("vtec")
        if want_event and want_event == ev:
            score += 2
            reasons.append("event")
        if want_sent and want_sent == sent:
            score += 2
            reasons.append("sent")

        if score > 0 and (best is None or score > best[0]):
            best = (score, props, vb)
            best_reason = ",".join(reasons)

    if not best:
        print("\n==================== CAP ====================")
        print(f"cap_url: {cap_url}")
        print(f"[WARN] No matching CAP feature found (it may have expired already).")
        print("Try running this DURING the alert window, or loosen matching:")
        print("  --cap-match-vtec ''  (disable vtec match)")
        print("  --cap-match-sent ''  (disable sent match)")
        print("  --cap-match-event '' (disable event match)")
        return 0

    _score, props, vb = best

    cap_script = build_cap_full_script(props)

    print("\n==================== CAP MATCH ====================")
    print(f"cap_url: {cap_url}")
    print(f"match_reason: {best_reason}")
    print(f"event: {props.get('event')}")
    print(f"sent: {props.get('sent')}")
    print(f"headline: {props.get('headline')}")
    print(f"areaDesc: {props.get('areaDesc')}")
    if vb:
        print(f"VTEC: {vb}")

    print("\n==================== CAP built script ====================")
    print(cap_script)
    find_markers("CAP script", cap_script)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
