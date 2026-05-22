# samegen

`samegen` is the optional native SAME/EAS AFSK WAV encoder for SeasonalWeather.
The Python encoder remains the safe fallback. SeasonalWeather only uses this
binary when `same.native_encoder.enabled` is true and the configured binary is
available.

Build locally:

```bash
cd tools/samegen
cargo build --release
sudo install -m 0755 target/release/samegen /usr/local/bin/samegen
```

Example:

```bash
samegen encode \
  --message 'ZCZC-WXR-RWT-024033+0015-1421345-SEASNWXR-' \
  --out /tmp/same.wav \
  --sample-rate 48000 \
  --amplitude 0.35 \
  --bursts 3 \
  --pause-seconds 1.0
```
