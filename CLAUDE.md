# Metaculus Forecasting Bot Template

This is a template for a forecasting bot designed to compete in Metaculus AI Forecasting Tournaments. The repository contains tools for making predictions on various questions using AI and search providers.

## Key Components

1. `SuperForecaster` class (`main.py`):
   - Uses superforecasting principles (reference classes, base rates, etc.)
   - Implements robust research gathering from multiple sources
   - Handles different question types (binary, numeric, multiple choice)
   - Uses OpenRouter's Perplexity model for predictions by default

2. Benchmark functionality (`community_benchmark.py`):
   - Tests different bot configurations against community predictions
   - Compares various parameter settings (temperature, prediction counts)
   - Calculates expected baseline scores for each configuration

3. Environment setup:
   - Uses API keys for search providers (stored in `.env`)
   - Connects to Metaculus API for retrieving questions and submitting forecasts
   - Requires METACULUS_TOKEN and at least one search provider key

## Running Options

- Test mode: `poetry run python main.py --mode test_questions`
- Tournament mode: `poetry run python main.py --mode tournament`
- Quarterly Cup mode: `poetry run python main.py --mode quarterly_cup`
- Benchmark: `poetry run python community_benchmark.py --mode run`

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:
- forecasting-tools package (handles API calls, question formats)
- Various search provider APIs (Perplexity, Exa, AskNews)
- OpenRouter integration for LLM access

## Typical Workflow

1. The bot retrieves questions from Metaculus
2. For each question, it conducts research using search APIs
3. It applies superforecasting techniques to generate predictions
4. It submits these predictions to Metaculus (if configured)
5. Results can be benchmarked against community forecasts