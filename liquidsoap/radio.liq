# SeasonalWeather radio automation (Liquidsoap)
# - cycle plane: request.queue(id="cycle")
# - interrupt plane: request.queue(id="alert") (higher priority)

set("log.stdout", false)
set("log.file", true)
set("log.file.path", "/var/log/seasonalweather/liquidsoap.log")

set("server.telnet", true)
set("server.telnet.bind_addr", "127.0.0.1")
set("server.telnet.port", 1234)

# Queues are controlled by the Python orchestrator via telnet:
#   alert.push /path/to/file.wav
#   cycle.push /path/to/file.wav
#   cycle.flush
#   alert.flush
cycle = request.queue(id="cycle")
alert = request.queue(id="alert")

# Always have something to output (silence) so the mount stays up
silence = blank()

radio = fallback(track_sensitive=false, [alert, cycle, silence])

output.icecast(
  %vorbis(samplerate=48000, channels=2, quality=0.5),
  host="127.0.0.1",
  port=8000,
  password="seasonal-source",
  mount="seasonalweather.ogg",
  name="SeasonalWeather",
  description="Unofficial automated IP weather broadcast (not NOAA Weather Radio)",
  genre="Weather",
  radio
)
