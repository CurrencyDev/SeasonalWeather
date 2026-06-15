from pathlib import Path


def test_radio_liq_uses_flat_priority_fallback_with_compatible_queue_syntax():
    script = Path("liquidsoap/radio.liq").read_text()

    assert 'cycle = request.queue(id="cycle")' in script
    assert 'voice_alert = request.queue(id="voice_alert")' in script
    assert 'full_alert = request.queue(id="full_alert")' in script
    assert "conservative=" not in script
    assert "alerts = fallback" not in script
    assert 'radio = fallback(track_sensitive=false, [full_alert, voice_alert, cycle, silence])' in script
    assert 'server.register(namespace="sw.cycle", usage="push <uri>"' in script
    assert 'server.register(namespace="sw.voice_alert", usage="push <uri>"' in script
    assert 'server.register(namespace="sw.full_alert", usage="push <uri>"' in script
    assert 'server.register(namespace="sw.cycle", usage="skip"' in script
    assert 'server.register(namespace="sw.voice_alert", usage="skip"' in script
    assert 'server.register(namespace="sw.full_alert", usage="skip"' in script
    assert 'usage="<uri>"' not in script
    assert 'usage=""' not in script
    assert 'description="Push voice alert audio and nudge cycle playback."' not in script
    assert 'description="Push full alert audio and nudge cycle playback."' not in script
    voice_push = script.split('def sw_voice_alert_push(uri) =', 1)[1].split('end', 1)[0]
    full_push = script.split('def sw_full_alert_push(uri) =', 1)[1].split('end', 1)[0]
    assert 'source.skip(cycle)' not in voice_push
    assert 'source.skip(cycle)' not in full_push
