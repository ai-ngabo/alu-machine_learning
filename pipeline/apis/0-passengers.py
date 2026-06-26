#!/usr/bin/env python3
"""
Module to interact with SWAPI and filter starships by passenger capacity.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of starships that can hold at least passengerCount passengers.
    Handles SWAPI pagination automatically.
    """
    url = "https://swapi.dev"
    matching_ships = []

    while url:
        try:
            response = requests.get(url).json()
        except Exception:
            break

        for ship in response.get('results', []):
            passengers_str = ship.get('passengers', '')
            
            # Remove commas often present in numbers like "1,000"
            passengers_str = passengers_str.replace(',', '')

            # Check if the value is a valid integer before comparing
            if passengers_str.isdigit():
                if int(passengers_str) >= passengerCount:
                    matching_ships.append(ship['name'])

        # Move to the next page of results
        url = response.get('next')

    return matching_ships

