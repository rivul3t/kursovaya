from typing import Dict, List, Any
import asyncio
import httpx

from config import OVERPASS_URL, REQUEST_TIMEOUT, REQUEST_RETRIES
from main import logger

OSM_TAG_MAP = {
    "bus_stop": ('node', 'highway', 'bus_stop'),
    "bridge": ('way', 'bridge', 'yes'),
    "river": ('way', 'waterway', 'river'),
    "forest": ('way', 'landuse', 'forest'),
    "traffic_signals": ('node', 'highway', 'traffic_signals'),
    "crossing": ('node', 'highway', 'crossing'),
    "fuel": ('node', 'amenity', 'fuel'),
}


async def overpass_query(query: str) -> Dict[str, Any]:
    """
    Execute an async Overpass API query.

    Args:
        query: Overpass QL string.

    Returns:
        Parsed JSON response or {"elements": []} if an error occurs.
    """
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            logger.info("Sending Overpass request (attempt %d)", attempt)

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    OVERPASS_URL,
                    data={"data": query}
                )

            logger.info("Overpass status code: %s", response.status_code)

            if response.status_code != 200:
                logger.error("Non-200 Overpass response")
                logger.debug("Response preview: %s", response.text[:300])
                continue

            return response.json()

        except httpx.TimeoutException:
            logger.warning("Timeout on attempt %d", attempt)

        except httpx.RequestError:
            logger.exception("Network error during Overpass request")

        except ValueError:
            logger.exception("JSON parsing failed")

        except Exception:
            logger.exception("Unexpected error in overpass_query")

        await asyncio.sleep(1.5 * attempt)

    logger.error("All Overpass retries failed")
    return {"elements": []}

def build_overpass_query(city: str, tags: list[str]) -> str:
    parts = []

    for tag in tags:
        if tag not in OSM_TAG_MAP:
            continue

        obj_type, key, value = OSM_TAG_MAP[tag]

        parts.append(
            f'{obj_type}["{key}"="{value}"](area.city);'
        )

    tag_query = "\n".join(parts)

    query = f"""
    [out:json][timeout:25];
    area["name"="{city}"]->.city;
    (
        {tag_query}
    );
    out center 50;
    """

    return query

async def find_nodes_in_city(
    city: str,
    tags: Dict[str, List[str]],
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Find OSM nodes in a given city using Overpass API.

    Args:
        city: City name.
        tags: Tag filters. Example: {"amenity": ["restaurant"]}
        limit: Max number of nodes.

    Returns:
        List of OSM elements.
    """
    try:
        logger.info("Building query for city: %s", city)

        if not tags:
            logger.warning("No tags provided")
            return []


        query = build_overpass_query(city, tags)

        logger.debug("Generated Overpass query: %s", query)

        result = await overpass_query(query)
        elements = result.get("elements", [])

        logger.info("Found %d elements", len(elements))

        return elements

    except Exception:
        logger.exception("Error in find_nodes_in_city")
        return []
