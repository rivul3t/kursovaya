find_nodes_in_city_tool = {
        "name": "find_nodes_in_city",
        "description": "Find likely OSM objects mentioned by the user",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "fuel",
                            "bridge",
                            "forest",
                            "river",
                            "traffic_signals",
                            "bus_stop",
                            "crossing",
                            "highway",
                            "curve"
                        ]
                    }
                }
            },
            "required": ["city", "tags"]
        }
    }