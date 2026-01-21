"""
Entry-point wrapper for the SeasonalWeather inject/debug tool.

- `python -m seasonalweather.cli.inject ...` works (calls main)
- future packaging can point console_scripts at seasonalweather.cli.inject:main
"""
import runpy


def main() -> None:
    # Run the copied tool module as if it were executed as a script.
    runpy.run_module("seasonalweather.cli.inject_tool", run_name="__main__")


if __name__ == "__main__":
    main()
