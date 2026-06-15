from pathlib import Path

from seasonalweather.liquidsoap_telnet import LiquidsoapTelnet


def _wire_fake_send(tn: LiquidsoapTelnet, help_text: str):
    commands: list[str] = []

    def fake_send(command: str, *, read_deadline=None):
        commands.append(command)
        if command == "help":
            return help_text
        return "OK\nEND\n"

    tn._send = fake_send  # type: ignore[method-assign]
    return commands


def test_liquidsoap_telnet_prefers_seasonalweather_aliases(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| sw.cycle.push <uri>
| sw.cycle.skip
| sw.voice_alert.push <uri>
| sw.voice_alert.skip
| sw.full_alert.push <uri>
| sw.full_alert.skip
| request_queue.push <uri>
| request_queue.1.push <uri>
| request_queue.2.push <uri>
END
""",
    )

    tn.push_full_alert(str(wav))
    tn.push_voice_alert(str(wav))
    tn.push_cycle(str(wav))
    tn.flush_alert()

    assert any(cmd.startswith("sw.full_alert.push ") for cmd in commands)
    assert any(cmd.startswith("sw.voice_alert.push ") for cmd in commands)
    assert any(cmd.startswith("sw.cycle.push ") for cmd in commands)
    assert "sw.full_alert.skip" in commands
    assert "sw.voice_alert.skip" in commands


def test_liquidsoap_telnet_uses_split_alert_planes(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| cycle.push <uri>
| cycle.flush
| voice_alert.push <uri>
| voice_alert.flush
| voice_alert.skip
| full_alert.push <uri>
| full_alert.flush
| full_alert.skip
END
""",
    )

    tn.push_full_alert(str(wav))
    tn.push_voice_alert(str(wav))
    tn.flush_alert()

    assert any(cmd.startswith("full_alert.push ") for cmd in commands)
    assert any(cmd.startswith("voice_alert.push ") for cmd in commands)
    assert "full_alert.flush" in commands
    assert "voice_alert.flush" in commands


def test_liquidsoap_telnet_collapses_to_legacy_alert_plane(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| cycle.push <uri>
| cycle.flush
| alert.push <uri>
| alert.flush
| alert.skip
END
""",
    )

    tn.push_full_alert(str(wav))
    tn.push_voice_alert(str(wav))
    tn.flush_alert()

    assert sum(1 for cmd in commands if cmd.startswith("alert.push ")) == 2
    assert commands.count("alert.flush") == 1


def test_liquidsoap_telnet_maps_anonymous_queues_by_declaration_order(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| request_queue.push <uri>
| request_queue.flush_and_skip
| request_queue.skip
| request_queue.1.push <uri>
| request_queue.1.flush_and_skip
| request_queue.1.skip
| request_queue.2.push <uri>
| request_queue.2.flush_and_skip
| request_queue.2.skip
END
""",
    )

    tn.push_cycle(str(wav))
    tn.push_voice_alert(str(wav))
    tn.push_full_alert(str(wav))

    assert any(cmd.startswith("request_queue.push ") for cmd in commands)
    assert any(cmd.startswith("request_queue.1.push ") for cmd in commands)
    assert any(cmd.startswith("request_queue.2.push ") for cmd in commands)

    cycle_cmd = next(cmd for cmd in commands if cmd.startswith("request_queue.push "))
    voice_cmd = next(cmd for cmd in commands if cmd.startswith("request_queue.1.push "))
    full_cmd = next(cmd for cmd in commands if cmd.startswith("request_queue.2.push "))
    assert cycle_cmd != voice_cmd != full_cmd


def test_liquidsoap_telnet_detects_aliases_without_exact_usage_suffix(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| sw.cycle.push : Push routine cycle audio.
| sw.cycle.skip
| sw.voice_alert.push <request-uri> : Push voice alert audio.
| sw.voice_alert.skip
| sw.full_alert.push <request-uri> : Push full alert audio.
| sw.full_alert.skip
| request_queue.push <uri>
| request_queue.1.push <uri>
| request_queue.2.push <uri>
END
""",
    )

    tn.push_full_alert(str(wav))
    tn.push_voice_alert(str(wav))
    tn.push_cycle(str(wav))

    assert any(cmd.startswith("sw.full_alert.push ") for cmd in commands)
    assert any(cmd.startswith("sw.voice_alert.push ") for cmd in commands)
    assert any(cmd.startswith("sw.cycle.push ") for cmd in commands)
    assert not any(cmd.startswith("request_queue") and ".push " in cmd for cmd in commands)


def test_liquidsoap_telnet_sends_bare_file_uri_to_sw_aliases(tmp_path: Path) -> None:
    wav = tmp_path / "alert with spaces.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| sw.cycle.push <uri>
| sw.cycle.skip
| sw.voice_alert.push <uri>
| sw.voice_alert.skip
| sw.full_alert.push <uri>
| sw.full_alert.skip
END
""",
    )

    tn.push_full_alert(str(wav), meta={"title": "Manual alert with spaces", "album": "Weather information"})

    push_cmd = next(cmd for cmd in commands if cmd.startswith("sw.full_alert.push "))
    assert "annotate:" not in push_cmd
    assert push_cmd.startswith("sw.full_alert.push file://")
    assert "%20" in push_cmd


def test_liquidsoap_telnet_keeps_annotations_for_builtin_request_queues(tmp_path: Path) -> None:
    wav = tmp_path / "alert.wav"
    wav.write_bytes(b"RIFFfakeWAVE")
    tn = LiquidsoapTelnet("127.0.0.1", 1234)
    commands = _wire_fake_send(
        tn,
        """
| request_queue.push <uri>
| request_queue.flush_and_skip
END
""",
    )

    tn.push_cycle(str(wav), meta={"title": "Cycle title"})

    push_cmd = next(cmd for cmd in commands if cmd.startswith("request_queue.push "))
    assert "annotate:" in push_cmd
    assert 'title="Cycle title"' in push_cmd
