from seasonalweather.worker.handlers import synthesize


def dispatch() -> None:
    synthesize()
