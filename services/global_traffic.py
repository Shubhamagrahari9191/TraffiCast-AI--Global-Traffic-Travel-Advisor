"""
Global Traffic & Travel Decision Engine
Handles:
1. Global Geocoding & Address Autocomplete (Nominatim / Photon / Cached Presets)
2. Road Network Routing (OSRM / Great-Circle Fallback)
3. Location & Timezone-Aware Spatiotemporal Traffic Flow Simulation
4. Multi-Window Departure Time Optimization & Travel Decision Advisory
"""

import math
import json
import time
import datetime
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional

# Popular global metropolitan presets for instant lookup
POPULAR_CITIES = [
    {"name": "New York, USA", "lat": 40.7128, "lng": -74.0060, "country": "United States", "tz_offset": -5.0, "peak_morning": 8.25, "peak_evening": 17.75, "default_dest": {"name": "JFK Airport, NY", "lat": 40.6413, "lng": -73.7781}},
    {"name": "London, UK", "lat": 51.5074, "lng": -0.1278, "country": "United Kingdom", "tz_offset": 0.0, "peak_morning": 8.00, "peak_evening": 17.50, "default_dest": {"name": "Heathrow Airport, London", "lat": 51.4700, "lng": -0.4543}},
    {"name": "Tokyo, Japan", "lat": 35.6762, "lng": 139.6503, "country": "Japan", "tz_offset": 9.0, "peak_morning": 8.50, "peak_evening": 18.25, "default_dest": {"name": "Haneda Airport, Tokyo", "lat": 35.5494, "lng": 139.7798}},
    {"name": "Paris, France", "lat": 48.8566, "lng": 2.3522, "country": "France", "tz_offset": 1.0, "peak_morning": 8.20, "peak_evening": 18.00, "default_dest": {"name": "Charles de Gaulle Airport, Paris", "lat": 49.0097, "lng": 2.5479}},
    {"name": "Mumbai, India", "lat": 19.0760, "lng": 72.8777, "country": "India", "tz_offset": 5.5, "peak_morning": 9.00, "peak_evening": 19.00, "default_dest": {"name": "Navi Mumbai, India", "lat": 19.0330, "lng": 73.0297}},
    {"name": "Los Angeles, USA", "lat": 34.0522, "lng": -118.2437, "country": "United States", "tz_offset": -8.0, "peak_morning": 8.00, "peak_evening": 17.25, "is_benchmark": "METR-LA", "default_dest": {"name": "Santa Monica, CA", "lat": 34.0195, "lng": -118.4912}},
    {"name": "San Francisco, USA", "lat": 37.7749, "lng": -122.4194, "country": "United States", "tz_offset": -8.0, "peak_morning": 8.25, "peak_evening": 17.50, "is_benchmark": "PEMS-BAY", "default_dest": {"name": "San Jose, CA", "lat": 37.3382, "lng": -121.8863}},
    {"name": "Dubai, UAE", "lat": 25.2048, "lng": 55.2708, "country": "United Arab Emirates", "tz_offset": 4.0, "peak_morning": 8.00, "peak_evening": 18.50, "default_dest": {"name": "Dubai Marina", "lat": 25.0805, "lng": 55.1403}},
    {"name": "Singapore", "lat": 1.3521, "lng": 103.8198, "country": "Singapore", "tz_offset": 8.0, "peak_morning": 8.30, "peak_evening": 18.00, "default_dest": {"name": "Changi Airport, Singapore", "lat": 1.3644, "lng": 103.9915}},
    {"name": "Berlin, Germany", "lat": 52.5200, "lng": 13.4050, "country": "Germany", "tz_offset": 1.0, "peak_morning": 7.75, "peak_evening": 17.00, "default_dest": {"name": "Berlin Brandenburg Airport", "lat": 52.3667, "lng": 13.5033}},
    {"name": "Sydney, Australia", "lat": -33.8688, "lng": 151.2093, "country": "Australia", "tz_offset": 10.0, "peak_morning": 8.00, "peak_evening": 17.30, "default_dest": {"name": "Sydney Airport", "lat": -33.9399, "lng": 151.1753}},
    {"name": "Toronto, Canada", "lat": 43.6532, "lng": -79.3832, "country": "Canada", "tz_offset": -5.0, "peak_morning": 8.15, "peak_evening": 17.45, "default_dest": {"name": "Pearson Airport, Toronto", "lat": 43.6777, "lng": -79.6248}}
]

GEOCODE_CACHE = {}

def estimate_timezone_offset(lat: float, lng: float) -> float:
    """Estimates local timezone offset (UTC +/- hours) based on longitude or known city lookup."""
    for city in POPULAR_CITIES:
        if abs(city["lat"] - lat) < 1.0 and abs(city["lng"] - lng) < 1.0:
            return city["tz_offset"]
    # Solar time longitude approximation (15 deg = 1 hour)
    return round((lng / 15.0) * 2) / 2.0

def geocode_address(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Geocodes a search string using OpenStreetMap Nominatim with caching and preset fallbacks."""
    if not query or len(query.strip()) < 2:
        return []
        
    query_clean = query.strip().lower()
    if query_clean in GEOCODE_CACHE:
        return GEOCODE_CACHE[query_clean]
        
    preset_matches = []
    for city in POPULAR_CITIES:
        if query_clean in city["name"].lower() or query_clean in city["country"].lower():
            preset_matches.append({
                "display_name": city["name"],
                "lat": city["lat"],
                "lng": city["lng"],
                "type": "city",
                "is_preset": True,
                "is_benchmark": city.get("is_benchmark")
            })
            
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&addressdetails=1&limit={limit}"
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'TraffiCast-AI-TrafficPrediction-Research/1.0'}
    )
    
    results = []
    try:
        with urllib.request.urlopen(req, timeout=3.5) as response:
            data = json.loads(response.read().decode())
            for item in data:
                results.append({
                    "display_name": item.get("display_name"),
                    "lat": float(item.get("lat")),
                    "lng": float(item.get("lon")),
                    "type": item.get("type", "location"),
                    "importance": item.get("importance", 0.5)
                })
    except Exception:
        results = preset_matches
        
    if not results and preset_matches:
        results = preset_matches

    GEOCODE_CACHE[query_clean] = results
    return results

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def fetch_osrm_route(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> Dict[str, Any]:
    """Fetches real road network route geometry and free-flow duration from OSRM public API."""
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}?"
           f"overview=full&geometries=geojson&steps=true")
    
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'TraffiCast-AI-TrafficPrediction-Research/1.0'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=4.0) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                distance_km = route["distance"] / 1000.0
                duration_min = route["duration"] / 60.0
                coordinates = route["geometry"]["coordinates"]
                lat_lngs = [[pt[1], pt[0]] for pt in coordinates]
                return {
                    "success": True,
                    "distance_km": round(distance_km, 2),
                    "free_flow_duration_min": round(duration_min, 1),
                    "coordinates": lat_lngs,
                    "provider": "OSRM"
                }
    except Exception:
        pass
        
    # Fallback: Great circle interpolation
    crow_dist = haversine_distance_km(origin_lat, origin_lng, dest_lat, dest_lng)
    road_dist = crow_dist * 1.28
    free_flow_speed = 55.0
    duration_min = (road_dist / free_flow_speed) * 60.0
    
    num_pts = 20
    lat_lngs = []
    for i in range(num_pts + 1):
        fraction = i / num_pts
        lat = origin_lat + fraction * (dest_lat - origin_lat)
        lng = origin_lng + fraction * (dest_lng - origin_lng)
        lat_lngs.append([round(lat, 5), round(lng, 5)])
        
    return {
        "success": True,
        "distance_km": round(road_dist, 2),
        "free_flow_duration_min": round(duration_min, 1),
        "coordinates": lat_lngs,
        "provider": "Interpolated-Fallback"
    }

def simulate_24h_traffic_profile(
    origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float,
    base_distance_km: float, base_duration_min: float, day_type: str = "weekday"
) -> Dict[str, Any]:
    """
    Simulates location-dependent 24-hour traffic congestion dynamics.
    Incorporates:
    - Origin/Destination coordinate geometry & road distance characteristics
    - Local timezone and local time calculation
    - Specific morning, midday, and evening peak shifts based on city typology
    - Contextual departure recommendations (Best Morning, Best Midday, Best Daytime overall)
    """
    time_slots = []
    durations = []
    speeds = []
    congestion_indices = []
    congestion_labels = []
    
    # 1. Estimate Local Timezone and Local Current Hour
    tz_offset = estimate_timezone_offset(origin_lat, origin_lng)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_now_dt = utc_now + datetime.timedelta(hours=tz_offset)
    local_hour_float = local_now_dt.hour + local_now_dt.minute / 60.0
    local_time_str = f"{local_now_dt.hour:02d}:{local_now_dt.minute:02d}"

    # 2. Location & Route Dynamics Fingerprinting
    # Urban density factor: shorter trips (< 15km) in cities have higher congestion delays than highways
    is_short_urban = base_distance_km < 20.0
    is_long_transit = base_distance_km > 60.0
    
    # Peak shift hash from coordinates (makes each unique route have realistic localized peak hours)
    coord_seed = (abs(origin_lat * 100) + abs(origin_lng * 100) + abs(dest_lat * 100) + abs(dest_lng * 100)) % 1.0
    morning_peak_hour = 7.75 + 1.0 * coord_seed  # Shifts between 7:45 AM and 8:45 AM
    evening_peak_hour = 17.25 + 1.2 * coord_seed # Shifts between 5:15 PM and 6:30 PM
    lunch_dip_hour = 12.5 + 0.8 * coord_seed     # Shifts between 12:30 PM and 1:15 PM

    # Free-flow baseline speed (km/h)
    free_flow_speed = max(18.0, (base_distance_km / (base_duration_min / 60.0))) if base_duration_min > 0 else 55.0

    # 3. Simulate 48 Departure Slots (00:00 to 23:30)
    for slot_idx in range(48):
        hour = slot_idx * 0.5
        h_int = int(hour)
        m_int = int((hour - h_int) * 60)
        time_label = f"{h_int:02d}:{m_int:02d}"
        
        if day_type == "weekend":
            # Weekend curve: Smooth morning, late-morning rise, afternoon activity plateau
            weekend_morning = 0.38 * math.exp(-((hour - 12.0) ** 2) / 9.0)
            weekend_afternoon = 0.45 * math.exp(-((hour - 17.0) ** 2) / 12.0)
            base_noise = 0.03 * math.sin(hour * 0.7 + coord_seed)
            congestion_factor = max(0.04, min(0.68, weekend_morning + weekend_afternoon + base_noise))
        else:
            # Weekday Bimodal Curve: Sharp Morning Rush + Midday Valley + Heavy Evening Rush
            # Congestion intensity adjusted for road type & urban density
            morning_amp = 0.85 if is_short_urban else (0.65 if is_long_transit else 0.76)
            evening_amp = 0.92 if is_short_urban else (0.72 if is_long_transit else 0.84)
            
            m_width = 1.6 if is_short_urban else 2.2
            e_width = 2.2 if is_short_urban else 2.8
            
            morning_rush = morning_amp * math.exp(-((hour - morning_peak_hour) ** 2) / m_width)
            lunch_bump = 0.32 * math.exp(-((hour - lunch_dip_hour) ** 2) / 4.5)
            evening_rush = evening_amp * math.exp(-((hour - evening_peak_hour) ** 2) / e_width)
            
            # Night smooth baseline (0.02 - 0.08)
            night_base = 0.03 if (hour < 5.0 or hour > 22.5) else 0.10
            
            raw_factor = morning_rush + lunch_bump + evening_rush + night_base
            congestion_factor = max(0.03, min(0.96, raw_factor))
            
        # Non-linear speed degradation
        speed_ratio = max(0.22, 1.0 - 0.74 * (congestion_factor ** 1.35))
        predicted_speed = free_flow_speed * speed_ratio
        predicted_duration = (base_distance_km / predicted_speed) * 60.0
        
        if congestion_factor < 0.20:
            c_label = "Smooth / Free Flow"
            c_level = "low"
        elif congestion_factor < 0.48:
            c_label = "Moderate Traffic"
            c_level = "moderate"
        elif congestion_factor < 0.76:
            c_label = "Heavy Congestion"
            c_level = "heavy"
        else:
            c_label = "Severe Gridlock"
            c_level = "severe"
            
        time_slots.append(time_label)
        durations.append(round(predicted_duration, 1))
        speeds.append(round(predicted_speed, 1))
        congestion_indices.append(round(congestion_factor * 100, 1))
        congestion_labels.append({"label": c_label, "level": c_level})

    # 4. Multi-Window Practical Recommendations
    # Normal Daytime Active Range: 07:00 (slot 14) to 20:00 (slot 40)
    morning_range = [(i, durations[i]) for i in range(12, 24)]   # 06:00 to 11:30
    midday_range = [(i, durations[i]) for i in range(21, 32)]    # 10:30 to 15:30
    evening_range = [(i, durations[i]) for i in range(38, 46)]   # 19:00 to 22:30

    best_morning_idx, best_morning_dur = min(morning_range, key=lambda x: x[1])
    best_midday_idx, best_midday_dur = min(midday_range, key=lambda x: x[1])
    best_evening_idx, best_evening_dur = min(evening_range, key=lambda x: x[1])

    # Find peak hours
    max_duration = max(durations)
    worst_idx = durations.index(max_duration)
    worst_time = time_slots[worst_idx]

    # Best daytime departure is the best slot between 07:00 AM and 07:00 PM
    active_day_range = [(i, durations[i]) for i in range(14, 38)] # 07:00 to 18:30
    best_daytime_idx, best_daytime_duration = min(active_day_range, key=lambda x: x[1])
    optimal_time = time_slots[best_daytime_idx]
    
    time_saved_vs_peak = round(max_duration - best_daytime_duration, 1)

    # 5. Local Real-Time "Leave Now vs Later" Evaluation
    current_slot_idx = int(round(local_hour_float * 2)) % 48
    current_duration = durations[current_slot_idx]
    current_congestion = congestion_labels[current_slot_idx]
    
    # Calculate difference for +30m, +1h, +2h
    leave_30m = durations[(current_slot_idx + 1) % 48]
    leave_1h = durations[(current_slot_idx + 2) % 48]
    leave_2h = durations[(current_slot_idx + 4) % 48]

    # Smart textual advice
    m_peak_str = f"{int(morning_peak_hour):02d}:{int((morning_peak_hour%1)*60):02d} – {int(morning_peak_hour+1.5):02d}:{int(((morning_peak_hour+1.5)%1)*60):02d}"
    e_peak_str = f"{int(evening_peak_hour):02d}:{int((evening_peak_hour%1)*60):02d} – {int(evening_peak_hour+1.75):02d}:{int(((evening_peak_hour+1.75)%1)*60):02d}"

    leave_now_advice = {
        "local_time": local_time_str,
        "tz_offset_hours": tz_offset,
        "current_duration_min": current_duration,
        "current_congestion": current_congestion,
        "is_currently_peak": current_congestion["level"] in ["heavy", "severe"],
        "leave_in_30m_duration": leave_30m,
        "leave_in_1h_duration": leave_1h,
        "leave_in_2h_duration": leave_2h,
        "best_morning_time": time_slots[best_morning_idx],
        "best_morning_duration": best_morning_dur,
        "best_midday_time": time_slots[best_midday_idx],
        "best_midday_duration": best_midday_dur,
        "best_evening_time": time_slots[best_evening_idx],
        "best_evening_duration": best_evening_dur,
    }

    return {
        "time_slots": time_slots,
        "durations_min": durations,
        "speeds_kmh": speeds,
        "speeds_mph": [round(s * 0.621371, 1) for s in speeds],
        "congestion_indices": congestion_indices,
        "congestion_levels": congestion_labels,
        "optimal_departure_time": optimal_time,
        "optimal_duration_min": best_daytime_duration,
        "worst_departure_time": worst_time,
        "worst_duration_min": max_duration,
        "time_saved_min": time_saved_vs_peak,
        "peak_morning_window": m_peak_str,
        "peak_evening_window": e_peak_str,
        "best_offpeak_windows": [
            f"Morning Window: {time_slots[best_morning_idx]} ({best_morning_dur} min)",
            f"Midday Window: {time_slots[best_midday_idx]} ({best_midday_dur} min)",
            f"Evening Free-Flow: After {time_slots[best_evening_idx]} ({best_evening_dur} min)"
        ],
        "leave_now_advice": leave_now_advice
    }

def plan_global_route(
    origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float,
    origin_name: str = "Origin", dest_name: str = "Destination",
    day_type: str = "weekday"
) -> Dict[str, Any]:
    """
    Main entry point for Global Traffic Forecasting & Travel Decision.
    """
    route_info = fetch_osrm_route(origin_lat, origin_lng, dest_lat, dest_lng)
    
    distance_km = route_info["distance_km"]
    base_duration = route_info["free_flow_duration_min"]
    coordinates = route_info["coordinates"]
    
    traffic_profile = simulate_24h_traffic_profile(
        origin_lat, origin_lng, dest_lat, dest_lng,
        distance_km, base_duration, day_type
    )
    
    # Route segmenting with dynamic congestion coloring
    route_segments = []
    num_coords = len(coordinates)
    if num_coords > 1:
        step = max(1, num_coords // 3)
        route_segments.append({
            "segment": "Origin Arterial Corridor",
            "coordinates": coordinates[:step+1],
            "severity": "moderate" if traffic_profile["leave_now_advice"]["is_currently_peak"] else "free_flow"
        })
        route_segments.append({
            "segment": "Main Transit Highway",
            "coordinates": coordinates[step:2*step+1],
            "severity": "free_flow"
        })
        route_segments.append({
            "segment": "Destination Hub Approach",
            "coordinates": coordinates[2*step:],
            "severity": "heavy" if traffic_profile["leave_now_advice"]["is_currently_peak"] else "moderate"
        })

    benchmark_tag = None
    if 33.7 < origin_lat < 34.4 and -118.7 < origin_lng < -118.0:
        benchmark_tag = "METR-LA (Los Angeles 207-Sensor Deep Learning Grid)"
    elif 37.2 < origin_lat < 38.0 and -122.6 < origin_lng < -121.7:
        benchmark_tag = "PEMS-BAY / PEMS04 (California Bay Area Spatiotemporal Grid)"

    return {
        "success": True,
        "origin": {"name": origin_name, "lat": origin_lat, "lng": origin_lng},
        "destination": {"name": dest_name, "lat": dest_lat, "lng": dest_lng},
        "distance_km": distance_km,
        "distance_miles": round(distance_km * 0.621371, 2),
        "free_flow_duration_min": base_duration,
        "benchmark_tag": benchmark_tag,
        "coordinates": coordinates,
        "route_segments": route_segments,
        "traffic_profile": traffic_profile
    }
