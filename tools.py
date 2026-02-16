"""
Tools for EcoHome Energy Advisor Agent
"""

import glob as glob_module
import math
import os
import random
from datetime import datetime, timedelta
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()


@tool
def get_weather_forecast(location: str, days: int = 3) -> dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
    """
    days = max(1, min(days, 7))

    # Seed based on location + date for reproducible but varied results
    seed = hash(location + datetime.now().strftime("%Y-%m-%d"))
    rng = random.Random(seed)

    # Base temperature varies by location keyword
    base_temp = 18.0
    if any(w in location.lower() for w in ["phoenix", "miami", "houston", "dallas"]):
        base_temp = 30.0
    elif any(w in location.lower() for w in ["seattle", "portland", "minneapolis"]):
        base_temp = 10.0
    elif any(
        w in location.lower() for w in ["san francisco", "los angeles", "san diego"]
    ):
        base_temp = 20.0

    conditions_weights = [
        ("sunny", 0.40),
        ("partly_cloudy", 0.30),
        ("cloudy", 0.20),
        ("rainy", 0.10),
    ]
    condition_names = [c[0] for c in conditions_weights]
    condition_probs = [c[1] for c in conditions_weights]

    # Solar irradiance multipliers by condition
    irradiance_mult = {
        "sunny": 1.0,
        "partly_cloudy": 0.6,
        "cloudy": 0.3,
        "rainy": 0.1,
    }

    # Generate current conditions
    current_hour = datetime.now().hour
    current_condition = rng.choices(condition_names, weights=condition_probs, k=1)[0]
    hour_temp_offset = (
        5.0 * math.sin(math.pi * (current_hour - 6) / 12)
        if 6 <= current_hour <= 18
        else -3.0
    )
    current_temp = round(base_temp + hour_temp_offset + rng.uniform(-2, 2), 1)

    forecast = {
        "location": location,
        "forecast_days": days,
        "current": {
            "temperature_c": current_temp,
            "condition": current_condition,
            "humidity": rng.randint(30, 80),
            "wind_speed": round(rng.uniform(2, 25), 1),
        },
        "hourly": [],
    }

    # Generate hourly data for each day
    for day in range(days):
        day_condition = rng.choices(condition_names, weights=condition_probs, k=1)[0]
        day_rng = random.Random(seed + day)

        for hour in range(24):
            # Temperature: sinusoidal curve, cool at night, warm midday
            temp_offset = (
                7.0 * math.sin(math.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -4.0
            )
            temp = round(
                base_temp + temp_offset + day_rng.uniform(-2, 2) + day * 0.5, 1
            )

            # Condition can vary slightly hour to hour
            if day_rng.random() < 0.7:
                hour_condition = day_condition
            else:
                hour_condition = day_rng.choices(
                    condition_names, weights=condition_probs, k=1
                )[0]

            # Solar irradiance: bell curve peaking at noon, modulated by weather
            if 6 <= hour <= 19:
                solar_factor = math.sin(math.pi * (hour - 6) / 13)
                solar_irradiance = round(
                    1000
                    * solar_factor
                    * irradiance_mult[hour_condition]
                    * day_rng.uniform(0.85, 1.15),
                    1,
                )
            else:
                solar_irradiance = 0.0

            humidity = rng.randint(35, 85)
            wind_speed = round(day_rng.uniform(2, 20), 1)

            forecast["hourly"].append(
                {
                    "day": day,
                    "hour": hour,
                    "temperature_c": temp,
                    "condition": hour_condition,
                    "solar_irradiance": solar_irradiance,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                }
            )

    return forecast


@tool
def get_electricity_prices(date: str = None) -> dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    base_rate = 0.12  # $/kWh

    hourly_rates = []
    for hour in range(24):
        if hour >= 22 or hour < 6:
            # Off-peak: 10 PM - 6 AM
            rate = 0.08
            period = "off_peak"
            demand_charge = 0.0
        elif 16 <= hour < 21:
            # Peak: 4 PM - 9 PM
            rate = 0.25
            period = "peak"
            demand_charge = 0.05
        else:
            # Mid-peak: 6 AM - 4 PM, 9 PM - 10 PM
            rate = 0.12
            period = "mid_peak"
            demand_charge = 0.02

        hourly_rates.append(
            {
                "hour": hour,
                "rate": rate,
                "period": period,
                "demand_charge": demand_charge,
            }
        )

    prices = {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "base_rate": base_rate,
        "hourly_rates": hourly_rates,
        "summary": {
            "off_peak_rate": 0.08,
            "mid_peak_rate": 0.12,
            "peak_rate": 0.25,
            "off_peak_hours": "22:00-06:00",
            "mid_peak_hours": "06:00-16:00, 21:00-22:00",
            "peak_hours": "16:00-21:00",
        },
    }

    return prices


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": [],
        }

        for record in records:
            usage_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": record.consumption_kwh,
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_usd": record.cost_usd,
                }
            )

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records)
                / max(1, (end_dt - start_dt).days),
                2,
            ),
            "records": [],
        }

        for record in records:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(
                    sum(r.consumption_kwh for r in usage_records), 2
                ),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": "sunny" if generation_records else "unknown",
            },
        }

        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += (
                record.consumption_kwh
            )
            summary["usage"]["device_breakdown"][device]["cost_usd"] += (
                record.cost_usd or 0
            )
            summary["usage"]["device_breakdown"][device]["records"] += 1

        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            documents = []
            doc_paths = glob_module.glob("data/documents/*.txt")
            for doc_path in doc_paths:
                loader = TextLoader(doc_path)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )

        docs = vectorstore.similarity_search(query, k=max_results)

        results = {
            "query": query,
            "total_results": len(docs),
            "tips": [],
        }

        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high"
                    if i < 2
                    else "medium"
                    if i < 4
                    else "low",
                }
            )

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    )

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
