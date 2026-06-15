# SeasonalWeather radio automation (Liquidsoap)
# - cycle plane: request.queue(id="cycle")
# - voice interrupt plane: request.queue(id="voice_alert")
# - full interrupt plane: request.queue(id="full_alert") (highest priority)
# - alert bus: FULL > VOICE, then routine cycle, with non-track-sensitive fallback
# - output mounts: Ogg/Vorbis and MP3 Icecast streams

set("log.stdout", false)
set("log.file", true)
set("log.file.path", "/var/log/seasonalweather/liquidsoap.log")

set("server.telnet", true)
set("server.telnet.bind_addr", "127.0.0.1")
set("server.telnet.port", 1234)

# Internal request queues.  Liquidsoap 2.3.x may expose these built-in
# request.queue controls as anonymous request_queue* telnet commands instead of
# the requested ids below.  Do not depend on those generated names for runtime
# queue routing; explicit SeasonalWeather aliases are registered below.
cycle = request.queue(id="cycle")
voice_alert = request.queue(id="voice_alert")
full_alert = request.queue(id="full_alert")

# Explicit, stable command aliases for the Python orchestrator.  These avoid the
# request_queue/request_queue.N ambiguity seen on Liquidsoap 2.3.x builds.
def sw_cycle_push(uri) =
  cycle.push(request.create(uri))
  "OK"
end

def sw_voice_alert_push(uri) =
  voice_alert.push(request.create(uri))
  "OK"
end

def sw_full_alert_push(uri) =
  full_alert.push(request.create(uri))
  "OK"
end

def sw_cycle_skip(_) =
  source.skip(cycle)
  "OK"
end

def sw_voice_alert_skip(_) =
  source.skip(voice_alert)
  "OK"
end

def sw_full_alert_skip(_) =
  source.skip(full_alert)
  "OK"
end

server.register(namespace="sw.cycle", usage="push <uri>", description="Push routine cycle audio.", "push", sw_cycle_push)
server.register(namespace="sw.voice_alert", usage="push <uri>", description="Push voice alert audio.", "push", sw_voice_alert_push)
server.register(namespace="sw.full_alert", usage="push <uri>", description="Push full alert audio.", "push", sw_full_alert_push)
server.register(namespace="sw.cycle", usage="skip", description="Skip current cycle audio.", "skip", sw_cycle_skip)
server.register(namespace="sw.voice_alert", usage="skip", description="Skip current voice alert audio.", "skip", sw_voice_alert_skip)
server.register(namespace="sw.full_alert", usage="skip", description="Skip current full alert audio.", "skip", sw_full_alert_skip)

# Always have something to output (silence) so the mount stays up.
silence = blank()

# Keep the runtime graph deliberately flat for Liquidsoap 2.3.x compatibility.
# Priority remains explicit: FULL beats VOICE, VOICE beats cycle, and silence
# keeps Icecast mounted if every request queue is momentarily empty.
radio = fallback(track_sensitive=false, [full_alert, voice_alert, cycle, silence])

# Primary low-bandwidth web stream.
output.icecast(
  %vorbis(samplerate=48000, channels=2, quality=0.5),
  host="127.0.0.1",
  port=8000,
  password="seasonal-source",
  mount="seasonalweather.ogg",
  name="SeasonalWeather Ogg",
  description="Unofficial automated IP weather broadcast (not NOAA Weather Radio)",
  genre="Weather",
  radio
)

# Compatibility stream for clients that do not reliably support Ogg/Vorbis.
output.icecast(
  %mp3(bitrate=128, samplerate=48000),
  host="127.0.0.1",
  port=8000,
  password="seasonal-source",
  mount="seasonalweather.mp3",
  name="SeasonalWeather MP3",
  description="Unofficial automated IP weather broadcast (not NOAA Weather Radio)",
  genre="Weather",
  radio
)

