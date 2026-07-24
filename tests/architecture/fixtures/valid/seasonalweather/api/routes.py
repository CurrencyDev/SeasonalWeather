from seasonalweather.services import WeatherService


async def status(service: WeatherService) -> dict[str, str]:
    return await service.status()
