#!/usr/bin/env python3
"""
Module to fetch names of home planets of all species from SWAPI
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of home planets of all species.
    Handles SWAPI pagination and preserves API-defined order.
    """
    planets = []
    url = "https://swapi.dev"

    while url:
        try:
            response = requests.get(url).json()
        except Exception:
            break

        for species in response.get('results', []):
            homeworld_url = species.get('homeworld')

            if homeworld_url:
                try:
                    planet_res = requests.get(homeworld_url).json()
                    planet_name = planet_res.get('name')
                    if planet_name and planet_name not in planets:
                        planets.append(planet_name)
                except Exception:
                    pass
            else:
                if "unknown" not in planets:
                    planets.append("unknown")

        url = response.get('next')

    return planets

