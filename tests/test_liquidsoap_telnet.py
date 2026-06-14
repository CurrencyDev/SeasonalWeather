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
