#!/usr/bin/env python3
"""
Module to fetch names of home planets of sentient species from SWAPI
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of home planets of all sentient species.
    Handles SWAPI pagination and filters by classification/designation.
    """
    planets = []
    url = "https://swapi.dev"

    while url:
        try:
            response = requests.get(url).json()
        except Exception:
            break

        for species in response.get('results', []):
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            # Filter for sentient species
            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')

                if homeworld_url:
                    try:
                        # Fetch the planet details
                        planet_res = requests.get(homeworld_url).json()
                        planet_name = planet_res.get('name')
                        if planet_name and planet_name not in planets:
                            planets.append(planet_name)
                    except Exception:
                        pass
                else:
                    # Handle species with no homeworld listed
                    if "unknown" not in planets:
                        planets.append("unknown")

        # Move to the next page
        url = response.get('next')

    return planets

