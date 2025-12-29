# SeasonalWeather state machine (v0.9)

The system is intentionally simple and NWR-inspired:

## States
- **NORMAL**
  - Default mode.
  - Cycle refresh interval = `cycle.normal_interval_seconds` (default 300s).
  - Content = station ID + status + HWO summary + forecast highlights + observations.

- **HEIGHTENED**
  - Entered when a “tone-out” product is received (warnings/watches; configurable list).
  - Cycle refresh interval = `cycle.heightened_interval_seconds` (default 180s).
  - Stays in HEIGHTENED until `now >= heightened_until`.

## Inputs
- **NWWS-OI XMPP products** (primary trigger)
  - Parsed for WFO (KLWX) and product type (SVR/FFW/TOR/etc).
  - If product type is in `policy.toneout_product_types`, the product becomes an **interrupt**.

- **Timers**
  - Every refresh, the current cycle audio is rebuilt and re-queued.

## Outputs
- **Interrupt plane** (Liquidsoap `alert` queue)
  - Attention tone → spoken alert → EOM beep → post-alert silence.
  - Preempts by flushing the cycle queue right before pushing the alert.

- **Cycle plane** (Liquidsoap `cycle` queue)
  - One pre-rendered WAV at a time; the queue is flushed before pushing the latest cycle.

## Transitions
- NORMAL → HEIGHTENED:
  - On receipt of a tone-out product.
  - Sets `last_heightened_at = now` and `heightened_until = now + min_heightened_seconds`.

- HEIGHTENED → NORMAL:
  - When `now >= heightened_until` and no newer tone-out extended it.

## Notes
- The “tone-out” list is configurable. Start conservative (warnings only) and expand later.
- The service area filter uses SAME/FIPS codes from LWX transmitters (KEC-83, KHB-36, WXM-42, WXM-43).
