# SeasonalWeather state machines

SeasonalWeather now has three distinct runtime state machines. Keep them separate so alert routing, PNS policy, and service-health presentation do not contaminate each other.

## Broadcast mode state

### NORMAL
- Default mode.
- Cycle refresh interval = `cycle.normal_interval_seconds`.
- Content = station ID, optional health notice, status, HWO, outlook/synopsis/forecast/observations, and active cycle-only alerts.

### HEIGHTENED
- Entered when a tone-out product is received.
- Tone-out products are controlled by `policy.toneout_product_types`.
- Cycle refresh interval = `cycle.heightened_interval_seconds`.
- Stays in HEIGHTENED until `now >= heightened_until` unless another tone-out extends it.

### Broadcast transitions
- `NORMAL -> HEIGHTENED`: tone-out product airs.
- `HEIGHTENED -> NORMAL`: `heightened_until` expires.

## PNS audio policy state

Public Information Statement handling is driven by `pns:` in `config.yaml` and implemented in `seasonalweather/pns.py`.

PNS is treated as a broad container product, not as inherently broadcast-safe prose. The PNS state machine classifies each product and returns one of these actions:

- `audio`: configured subtype matched and the body passed coherence checks.
- `ui_only`: useful product, but not safe for spoken-cycle audio.
- `drop`: no useful coherent text could be synthesized.
- `no_match` / `stale` / `disabled`: no cycle audio.

Default audio-enabled subtypes:

- Severe Weather Safety Rules.
- NOAA Weather Radio transmitter outage.
- NOAA Weather Radio transmitter restoration.

Hard reject/downrank signals include tabular report structure, `METADATA`, `TIME/DATE` table headers, report tokens such as `PKGUST`, dense numeric blocks, aligned observation rows, and configured reject keywords such as `spotter reports`.

The delimiter line `&&` is treated as a hard spoken-text boundary before metadata-style content unless a future subtype explicitly handles that content.

## Health state machine

Source-health presentation is driven by `health:` in `config.yaml` and implemented in `seasonalweather/health_state.py`.

### NORMAL
- Required sources are fresh enough.
- No degraded notice is inserted.

### SOURCE_IMPAIRED
- A redundant alert source is impaired, but a primary alert source remains healthy.
- The cycle includes a reduced-redundancy service notice.

### DEGRADED
- One or more important data classes are stale or failing, but normal programming remains useful.
- The cycle includes a degraded-mode service notice.

### CRITICAL_DEGRADED
- A primary/critical alert source is impaired while other sources still exist.
- The cycle warns that watches, warnings, and advisories may be delayed or unavailable.

### DETACHED
- All enabled alert-feed paths are impaired.
- If `health.detached_loop_only` is true, normal programming is suppressed and the cycle becomes a clear service-unavailable notice until source health recovers.

Health transitions use failure thresholds, stale timers, and `health.min_hold_seconds` hysteresis so the cycle does not flap from a single transient HTTP/XMPP failure.
