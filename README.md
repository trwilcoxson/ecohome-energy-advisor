# EcoHome Energy Advisor

An AI-powered energy optimization agent built with LangGraph that helps smart-home customers reduce electricity costs and environmental impact through personalized, data-driven recommendations.

## Project Overview

EcoHome is a smart-home energy start-up that helps customers with solar panels, electric vehicles, and smart thermostats optimize their energy usage. The Energy Advisor agent uses a ReAct (Reasoning + Acting) architecture to provide personalized recommendations about when to run devices to minimize costs and carbon footprint.

### Key Features

- **Weather-Aware Optimization**: Forecasts solar generation and adjusts recommendations based on weather conditions
- **Time-of-Use Pricing**: Optimizes device scheduling around peak, mid-peak, and off-peak electricity rates
- **Historical Usage Analysis**: Queries 30 days of energy consumption data by device type for pattern-based advice
- **RAG Knowledge Base**: Retrieves from 7 curated documents covering HVAC, solar, automation, storage, and seasonal tips
- **Multi-Device Optimization**: Handles EVs, HVAC systems, appliances, pool pumps, and battery storage
- **Quantified Savings**: Calculates specific dollar amounts, kWh savings, and annual projections
- **LLM-as-Judge Evaluation**: Automated evaluation pipeline scoring accuracy, relevance, completeness, and usefulness
- **Visualization Dashboard**: Heatmaps, radar charts, and cost analysis charts for data-driven insights

## Architecture

```
User Question
    |
    v
[LangGraph ReAct Agent] -- GPT-4o-mini
    |
    +-- get_weather_forecast()      # Mock weather API with realistic hourly data
    +-- get_electricity_prices()    # Time-of-use pricing (peak/mid-peak/off-peak)
    +-- query_energy_usage()        # SQLite historical consumption data
    +-- query_solar_generation()    # SQLite solar production records
    +-- get_recent_energy_summary() # 24-hour usage/generation summary
    +-- search_energy_tips()        # ChromaDB RAG with 7 knowledge documents
    +-- calculate_energy_savings()  # Savings calculator with annual projections
    |
    v
Structured Response with specific times, dollar amounts, and kWh estimates
```

## Project Structure

```
ecohome_solution/
├── models/
│   ├── __init__.py
│   └── energy.py                   # SQLAlchemy models (EnergyUsage, SolarGeneration)
├── data/
│   └── documents/
│       ├── tip_device_best_practices.txt
│       ├── tip_energy_savings.txt
│       ├── tip_hvac_optimization.txt
│       ├── tip_smart_home_automation.txt
│       ├── tip_renewable_energy_integration.txt
│       ├── tip_seasonal_energy_management.txt
│       └── tip_energy_storage_optimization.txt
├── agent.py                        # LangGraph ReAct agent with error handling
├── tools.py                        # 7 agent tools (weather, pricing, DB, RAG, calculator)
├── requirements.txt                # Python dependencies
├── 01_db_setup.ipynb              # Database setup with 30 days of sample data
├── 02_rag_setup.ipynb             # RAG pipeline: load 7 docs → ChromaDB vectorstore
├── 03_run_and_evaluate.ipynb      # 12 test cases, LLM evaluation, visualizations
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Notebooks

Execute in order:

1. **01_db_setup.ipynb** — Creates SQLite database with 30 days of energy usage and solar generation data (2,160+ records per table)
2. **02_rag_setup.ipynb** — Loads all 7 knowledge documents, splits into chunks, creates ChromaDB vectorstore with OpenAI embeddings
3. **03_run_and_evaluate.ipynb** — Runs 12 test cases, evaluates with LLM-as-judge, generates report and visualizations

## Agent Tools

| Tool | Type | Description |
|------|------|-------------|
| `get_weather_forecast` | Mock API | Hourly temperature, conditions, solar irradiance for 1-7 days |
| `get_electricity_prices` | Mock API | 24-hour TOU pricing: off-peak $0.08, mid-peak $0.12, peak $0.25/kWh |
| `query_energy_usage` | Database | Historical consumption by date range and device type |
| `query_solar_generation` | Database | Solar production with weather correlation |
| `get_recent_energy_summary` | Database | Device breakdown for last N hours |
| `search_energy_tips` | RAG | Semantic search across 7 knowledge base documents |
| `calculate_energy_savings` | Calculator | Per-device savings with annual projections |

## Evaluation

The agent is evaluated on 12 diverse test scenarios using two methods:

**Response Quality (LLM-as-Judge):**
- Accuracy, Relevance, Completeness, Usefulness — each scored 1-10

**Tool Usage (Programmatic):**
- Appropriateness: % of tools used that were expected
- Completeness: % of expected tools that were used

## Stand-Out Features

- **7 knowledge base documents** covering HVAC, automation, renewables, seasonal management, and storage
- **12 comprehensive test cases** spanning EV charging, HVAC optimization, cost analysis, weekend planning, and battery storage
- **LLM-as-judge evaluation** with GPT-4o-mini scoring on 4 quality dimensions
- **3 matplotlib visualizations**: evaluation radar chart, energy usage heatmap, and cost optimization breakdown

## Key Technologies

- **LangChain + LangGraph**: ReAct agent with tool calling
- **ChromaDB**: Vector database for RAG document retrieval
- **OpenAI GPT-4o-mini**: Agent LLM and embeddings
- **SQLAlchemy + SQLite**: Energy data persistence
- **Matplotlib + Pandas**: Data visualization and analysis
