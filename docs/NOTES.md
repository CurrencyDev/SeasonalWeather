# NOTES.md

Developer and agent notes on known gaps, intentional deferrals, and non-obvious design decisions.
These are not bugs unless noted as such. They are context.

---

## Discord errors channel is wired but not instrumented

`DiscordLogger.error()` and `DiscordLogger.warning()` exist and work correctly.
The drain loop delivers payloads reliably. The errors webhook channel is properly configured.

However, as of the time of this note, `discord.error()` is called in exactly **one place**
in the entire codebase: the SAME generation failure handler in `main.py`. All other exception
sites (`~492` across the project) log only to Python's standard logger (`log.exception`,
`log.error`, `log.warning`), meaning they surface in journalctl only and never reach Discord.

**This is not a bug in `discord_log.py`.** The pipe works. It is simply not wired at exception sites.

**Implication for agents:** When touching an exception block that represents a meaningful
operational failure (not a routine parse skip or a recoverable retry), consider calling
`self.discord.error(...)` in addition to `log.exception(...)`. The appropriate channel is
the `errors` channel. See existing call at the SAME generation failure site for reference style.

**Implication for operators:** The Discord errors channel being empty does not mean the service
is error-free. It means the instrumentation coverage is low. Authoritative error output is
in journalctl (`journalctl -u seasonalweather.service`).

---
