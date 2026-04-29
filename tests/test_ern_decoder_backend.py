from seasonalweather.broadcast.ern_gwes import _same_listen_module_cmd


def test_samedec_decoder_command_uses_samedec_module_and_flags() -> None:
    cmd = _same_listen_module_cmd(
        "http://example.invalid/stream",
        sr=48000,
        dedupe=20.0,
        trigger_ratio=8.0,
        tail=10.0,
        decoder_backend="samedec",
        samedec_bin="/usr/local/bin/samedec",
        samedec_confidence=0.85,
        samedec_start_delay_s=1.4,
    )

    assert "seasonalweather.same.listen_samedec" in cmd
    assert "--samedec-bin" in cmd
    assert "--confidence" in cmd
    assert "--start-delay-s" in cmd


def test_native_decoder_command_omits_samedec_only_flags() -> None:
    cmd = _same_listen_module_cmd(
        "http://example.invalid/stream",
        sr=48000,
        dedupe=20.0,
        trigger_ratio=8.0,
        tail=10.0,
        decoder_backend="native",
        samedec_bin="/usr/local/bin/samedec",
        samedec_confidence=0.85,
        samedec_start_delay_s=1.4,
    )

    assert "seasonalweather.same.listen" in cmd
    assert "seasonalweather.same.listen_samedec" not in cmd
    assert "--samedec-bin" not in cmd
    assert "--confidence" not in cmd
    assert "--start-delay-s" not in cmd


def test_auto_decoder_falls_back_to_native_when_samedec_missing() -> None:
    cmd = _same_listen_module_cmd(
        "http://example.invalid/stream",
        sr=48000,
        dedupe=20.0,
        trigger_ratio=8.0,
        tail=10.0,
        decoder_backend="auto",
        samedec_bin="/definitely/missing/samedec",
        samedec_confidence=0.85,
        samedec_start_delay_s=1.4,
    )

    assert "seasonalweather.same.listen" in cmd
    assert "seasonalweather.same.listen_samedec" not in cmd
    assert "--samedec-bin" not in cmd
