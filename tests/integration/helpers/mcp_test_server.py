"""Standalone MCP test server for integration tests.

Exposes two tools:
- get_capital(country: str) -> str
- get_population(city: str) -> str

Run standalone: python mcp_test_server.py
"""

from __future__ import annotations

from orbiter.mcp import mcp_server  # pyright: ignore[reportMissingImports]

# Known capitals and populations for testing
_CAPITALS: dict[str, str] = {
    "japan": "Tokyo",
    "france": "Paris",
    "germany": "Berlin",
    "australia": "Canberra",
    "brazil": "Brasilia",
    "canada": "Ottawa",
    "india": "New Delhi",
    "china": "Beijing",
    "usa": "Washington D.C.",
    "united states": "Washington D.C.",
    "uk": "London",
    "united kingdom": "London",
    "italy": "Rome",
    "spain": "Madrid",
    "mexico": "Mexico City",
    "argentina": "Buenos Aires",
}

_POPULATIONS: dict[str, str] = {
    "tokyo": "approximately 14 million (city proper)",
    "paris": "approximately 2 million (city proper)",
    "berlin": "approximately 3.6 million",
    "canberra": "approximately 450,000",
    "brasilia": "approximately 3 million",
    "london": "approximately 9 million",
    "rome": "approximately 2.8 million",
    "madrid": "approximately 3.3 million",
    "dublin": "approximately 1.2 million",
    "sydney": "approximately 5.3 million",
    "new york": "approximately 8 million (city proper)",
    "osaka": "approximately 2.7 million",
}


@mcp_server(name="test-server")
class TestServer:
    """Test MCP server for integration tests."""

    def get_capital(self, country: str) -> str:
        """Return the capital city of the given country.

        Args:
            country: The name of the country.

        Returns:
            The capital city name or an informative message.
        """
        key = country.lower().strip()
        return _CAPITALS.get(key, f"Capital of {country} is not in the test database.")

    def get_population(self, city: str) -> str:
        """Return the approximate population of the given city.

        Args:
            city: The name of the city.

        Returns:
            The approximate population or an informative message.
        """
        key = city.lower().strip()
        return _POPULATIONS.get(key, f"Population of {city} is not in the test database.")


if __name__ == "__main__":
    server = TestServer()
    server.run(transport="stdio")  # type: ignore[attr-defined]
