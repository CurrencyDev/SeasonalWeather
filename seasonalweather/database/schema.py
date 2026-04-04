from __future__ import annotations

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS active_alerts (
            alert_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            event TEXT NOT NULL,
            code TEXT NOT NULL,
            headline TEXT NOT NULL,
            script_text TEXT NOT NULL,
            audio_path TEXT,
            expires_at TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            cycle_only INTEGER NOT NULL DEFAULT 0,
            watch_number INTEGER,
            first_aired_at TEXT,
            last_aired_at TEXT,
            airing_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_active_alerts_expires_at ON active_alerts (expires_at)",
        """
        CREATE TABLE IF NOT EXISTS active_alert_vtec (
            alert_id TEXT NOT NULL,
            vtec TEXT NOT NULL,
            PRIMARY KEY (alert_id, vtec),
            FOREIGN KEY (alert_id) REFERENCES active_alerts(alert_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS active_alert_same (
            alert_id TEXT NOT NULL,
            same_code TEXT NOT NULL,
            PRIMARY KEY (alert_id, same_code),
            FOREIGN KEY (alert_id) REFERENCES active_alerts(alert_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cap_seen_ledger (
            dedupe_key TEXT PRIMARY KEY,
            seen_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cap_seen_ledger_seen_at ON cap_seen_ledger (seen_at)",
        """
        CREATE TABLE IF NOT EXISTS api_commands (
            command_id TEXT PRIMARY KEY,
            command_type TEXT NOT NULL,
            status TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            actor TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            request_id TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            idempotent_replay_count INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            error_json TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_api_commands_accepted_at ON api_commands (accepted_at)",
        """
        CREATE TABLE IF NOT EXISTS cycle_segments (
            segment_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            duration_s REAL NOT NULL,
            last_updated_ts REAL NOT NULL,
            refresh_interval_s INTEGER NOT NULL,
            is_placeholder INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scheduler_state (
            scheduler_name TEXT PRIMARY KEY,
            last_run_at TEXT,
            next_run_at TEXT,
            state_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS audio_assets (
            asset_id TEXT PRIMARY KEY,
            wav_path TEXT NOT NULL,
            original_filename TEXT,
            content_type TEXT,
            sha256 TEXT,
            headline TEXT,
            event_code TEXT,
            actor TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            meta_json TEXT
        )
        """,
    ),
    2: (
        "CREATE INDEX IF NOT EXISTS idx_api_commands_status_finished_at ON api_commands (status, finished_at)",
        "CREATE INDEX IF NOT EXISTS idx_audio_assets_expires_at ON audio_assets (expires_at)",
    ),
}
