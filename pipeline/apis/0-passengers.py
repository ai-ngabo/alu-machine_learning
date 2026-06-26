#!/usr/bin/env python3
import requests

def availableShips(passengerCount):
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:  # loop through all pages
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship["name"])

        url = data.get("next")  # move to next page if available

    return ships

