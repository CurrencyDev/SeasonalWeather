# SeasonalWeather radio automation (Liquidsoap)
# - cycle plane: request.queue(id="cycle")
# - voice interrupt plane: request.queue(id="voice_alert")
# - full interrupt plane: request.queue(id="full_alert") (highest priority)
# - output mounts: Ogg/Vorbis and MP3 Icecast streams

set("log.stdout", false)
set("log.file", true)
set("log.file.path", "/var/log/seasonalweather/liquidsoap.log")

set("server.telnet", true)
set("server.telnet.bind_addr", "127.0.0.1")
set("server.telnet.port", 1234)

# Queues are controlled by the Python orchestrator via telnet:
#   full_alert.push /path/to/file.wav
#   voice_alert.push /path/to/file.wav
#   cycle.push /path/to/file.wav
#   cycle.flush
#   voice_alert.flush / voice_alert.skip
#   full_alert.flush / full_alert.skip
cycle = request.queue(id="cycle")
voice_alert = request.queue(id="voice_alert")
full_alert = request.queue(id="full_alert")

# Always have something to output (silence) so the mount stays up
silence = blank()

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

