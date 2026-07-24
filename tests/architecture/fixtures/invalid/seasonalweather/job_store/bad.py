from seasonalweather.worker.handlers import run_job


def violate_repository_boundary() -> None:
    run_job()
