from seasonalweather.discord_log import DiscordLogger


def _dequeue_payload(logger: DiscordLogger):
    _url, payload = logger._queue.get_nowait()  # type: ignore[attr-defined]
    return payload


def test_alert_updated_uses_event_color_and_icon_for_voice_lifecycle() -> None:
    logger = DiscordLogger(
        alerts_url="https://discord.example/alerts",
        alerts_enabled=True,
        ops_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        icon_cdn_url="https://icons.example",
    )

    logger.alert_updated(
        code="SVR",
        event="Severe Thunderstorm Warning",
        vtec_action="CON",
        source="NWWS-OI",
        area="Montgomery, MD",
        vtec=["/O.CON.KLWX.SV.W.0123.260614T2000Z-260614T2030Z/"],
    )

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["color"] == 0xFF8C00
    assert embed["title"] == "Severe Thunderstorm Warning — continuing"
    assert "icon=cloud-lightning" in embed["thumbnail"]["url"]
    assert "hex=FF8C00" in embed["thumbnail"]["url"]


def test_alert_expired_keeps_gray_terminal_color_but_uses_event_icon() -> None:
    logger = DiscordLogger(
        alerts_url="https://discord.example/alerts",
        alerts_enabled=True,
        ops_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        icon_cdn_url="https://icons.example",
    )

    logger.alert_expired(
        code="TOR",
        event="Tornado Warning",
        vtec_action="CAN",
        source="NWWS-OI",
        area="Fairfax, VA",
        vtec=["/O.CAN.KLWX.TO.W.0024.260709T1830Z-260709T1845Z/"],
    )

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["color"] == 0x888888
    assert embed["title"] == "Tornado Warning — cancelled"
    assert "icon=siren" in embed["thumbnail"]["url"]
    assert "hex=888888" in embed["thumbnail"]["url"]


def test_alert_partial_terminal_uses_event_color_and_track_fields() -> None:
    logger = DiscordLogger(
        alerts_url="https://discord.example/alerts",
        alerts_enabled=True,
        ops_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        icon_cdn_url="https://icons.example",
    )

    logger.alert_partial_terminal(
        code="SVA",
        event="Severe Thunderstorm Watch",
        vtec_action="CAN",
        source="NWWS-OI",
        area="Frederick, MD; Montgomery, MD",
        ended_tracks=["KLWX.SV.A.0474"],
        continuing_tracks=["KPHI.SV.A.0474"],
        vtec=[
            "/O.CAN.KLWX.SV.A.0474.260709T1830Z-260710T0200Z/",
            "/O.CON.KPHI.SV.A.0474.260709T1830Z-260710T0200Z/",
        ],
    )

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["color"] == 0xDB7093
    assert embed["title"] == "Severe Thunderstorm Watch — partially cancelled"
    assert "icon=cloud-lightning" in embed["thumbnail"]["url"]
    assert "hex=DB7093" in embed["thumbnail"]["url"]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["Mode"] == "Voice-only partial lifecycle (no retone)"
    assert "KLWX.SV.A.0474" in fields["Ended tracks"]
    assert "KPHI.SV.A.0474" in fields["Continuing tracks"]


def test_alert_decision_is_gated_by_ops_detail_log() -> None:
    logger = DiscordLogger(
        ops_url="https://discord.example/ops",
        ops_enabled=True,
        alerts_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        ops_detail_log=False,
    )

    logger.alert_decision(source="NWWS-OI", result="skip", reason="dedupe")
    assert logger._queue.empty()  # type: ignore[attr-defined]


def test_alert_decision_posts_structured_ops_embed_when_enabled() -> None:
    logger = DiscordLogger(
        ops_url="https://discord.example/ops",
        ops_enabled=True,
        alerts_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        ops_detail_log=True,
        icon_cdn_url="https://icons.example",
    )

    logger.alert_decision(
        source="NWWS-OI",
        result="air",
        reason="vtec_new_full",
        event="Severe Thunderstorm Warning",
        code="SVR",
        product_type="SVR",
        mode="full",
        awips="SVRLWX",
        wfo="KLWX",
        same_targets=3,
        zones=4,
        vtec=["/O.NEW.KLWX.SV.W.0123.260614T2000Z-260614T2030Z/"],
        details={"target_source": "ugc-zone", "mapped_ok": True},
    )

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["title"] == "NWWS-OI decision — air"
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["Result"] == "air"
    assert fields["Reason"] == "vtec_new_full"
    assert fields["SAME targets"] == "3"
    assert fields["Target Source"] == "ugc-zone"


def test_source_health_posts_when_enabled_by_default() -> None:
    logger = DiscordLogger(
        ops_url="https://discord.example/ops",
        ops_enabled=True,
        alerts_enabled=False,
        api_enabled=False,
        errors_enabled=False,
        icon_cdn_url="https://icons.example",
    )

    logger.source_health(
        source="NWWS-OI",
        status="enabled",
        severity="ok",
        details={"allowed_wfos": "KLWX", "toneout_products": 6},
    )

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["title"] == "NWWS-OI — enabled"
    assert embed["color"] == 0x639922
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["Allowed Wfos"] == "KLWX"


def test_audio_pipeline_failure_uses_errors_channel_even_without_detail_log() -> None:
    logger = DiscordLogger(
        errors_url="https://discord.example/errors",
        errors_enabled=True,
        alerts_enabled=False,
        ops_enabled=False,
        api_enabled=False,
        ops_detail_log=False,
    )

    logger.audio_pipeline(source="nwws-full", status="failed", mode="full", fallback="RuntimeError")

    embed = _dequeue_payload(logger)["embeds"][0]
    assert embed["title"] == "Audio pipeline — failed"
    assert embed["color"] == 0xE24B4A
