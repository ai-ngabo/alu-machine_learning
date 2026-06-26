#!/usr/bin/env python3
"""
Module for retrieving starships from the Star Wars API (SWAPI).

This module defines the function availableShips, which returns a list
of starships that can hold at least a given number of passengers.
Pagination is handled automatically. If no ships are available, an
empty list is returned.
"""

import requests


def availableShips(passengerCount):
    """
    Return starships that can hold at least passengerCount passengers.

    Args:
        passengerCount (int): Minimum number of passengers required.

    Returns:
        list: Names of starships meeting the requirement.
    """
    url = "https://swapi.dev/api/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship["name"])

        url = data.get("next")

    return ships

