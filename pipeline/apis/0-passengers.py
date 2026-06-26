#!/usr/bin/env python3
"""Module to filter SWAPI starships by passenger count."""
import requests


def availableShips(passengerCount):
    """Return list of ships matching or exceeding passengerCount."""
    url = "https://alx-tools.com"
    matching_ships = []

    while url:
        try:
            res = requests.get(url).json()
        except Exception:
            break

        for ship in res.get('results', []):
            p_str = ship.get('passengers', '')
            p_str = p_str.replace(',', '')

            if p_str.isdigit() and int(p_str) >= passengerCount:
                matching_ships.append(ship['name'])

        url = res.get('next')

    return matching_ships

