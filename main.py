import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class SuperForecaster(ForecastBot):
    """
    Simplified forecasting bot implementing superforecasting principles:
    - Uses reference class forecasting to ground predictions in base rates
    - Implements Tetlock's superforecasting principles: probabilistic thinking, belief updating
    - Streamlined for faster execution with reduced timeouts

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    General flow:
    - Load questions from Metaculus
    - For each question
        - Execute run_research once
        - Execute respective run_forecast function with fewer predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects
    """

    _max_concurrent_questions = 4  # Increased for parallel processing
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Simplified research function that prioritizes speed over comprehensiveness.
        Only uses OpenRouter for research to avoid timeouts.
        """
        async with self._concurrency_limiter:
            # Only use OpenRouter - fastest and most reliable option
            if os.getenv("OPENROUTER_API_KEY"):
                try:
                    openrouter_research = await self._call_perplexity(question.question_text, use_open_router=True)
                    if openrouter_research:
                        logger.info(f"Found Research for URL {question.page_url}")
                        return f"--- OPENROUTER SEARCH RESULTS ---\n{openrouter_research}"
                except Exception as e:
                    logger.warning(f"Error using OpenRouter research for {question.page_url}: {str(e)}")
            
            logger.warning(f"No research provider found when processing question URL {question.page_url}. Will pass back empty string.")
            return ""

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        Enhanced SmartSearcher implementation that does a more thorough job with Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=3,  # Increased from 2
            num_sites_per_search=12,  # Increased from 10
        )
        
        # Enhanced prompt with focus on superforecasting principles
        prompt = (
            "You are a research assistant to a superforecaster. The superforecaster will give "
            "you a question they intend to forecast on. To be a great assistant, you must: "
            "1. Generate a comprehensive and factual rundown of relevant information "
            "2. Find key statistics, historical precedents, and expert opinions "
            "3. Identify recent developments that might change previous trends "
            "4. Look for both confirming and disconfirming evidence "
            "5. Avoid biases in your research and reporting "
            "6. Focus on finding solid reference classes for base rates "
            f"\n\nThe question is: {question}"
        )
        
        # Get both recent and historical results
        try:
            # Default search (without date filters)
            recent_response = await searcher.invoke(prompt + "\nPlease focus on the most recent developments.")
            historical_response = await searcher.invoke(prompt + "\nPlease focus on historical context and precedents.")
            
            # Combine results
            combined_response = f"RECENT DEVELOPMENTS:\n{recent_response}\n\nHISTORICAL CONTEXT:\n{historical_response}"
            
            return combined_response
        except Exception as e:
            logger.warning(f"Error in SmartSearcher.invoke: {str(e)}")
            return ""
        
    async def _summarize_multi_source_research(self, combined_research: str, question: str) -> str:
        """
        Summarizes research from multiple sources - shortened for speed
        """
        summarization_prompt = clean_indents(
            f"""
            You are a research analyst assisting a superforecaster.
            
            QUESTION: {question}
            
            Synthesize this research into a concise summary:
            1. Key facts and statistics (2-3 bullet points)
            2. Most relevant reference classes and base rates
            3. Areas of uncertainty (1-2 points)
            4. Current trend direction
            
            FORMAT: 
            - Key Facts (2-3 points)
            - Critical Data (numbers and dates only)
            - Reference Classes
            - Uncertainties
            
            RESEARCH:
            {combined_research}
            """
        )
        
        # Use a slightly higher temperature for summary to encourage synthesis
        summary_llm = self.get_llm("summarizer", "llm") if "summarizer" in self.llms else self.get_llm("default", "llm")
        summary = await summary_llm.invoke(summarization_prompt, temperature=0.3)
        
        return summary

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Enhanced binary forecasting model that implements superforecasting techniques:
        - Reference class reasoning
        - Base rate analysis
        - Outside view consideration
        - Pre-mortem analysis
        - Multiple perspective taking
        """
        prompt = clean_indents(
            f"""
            You are a superforecaster trained in Tetlock's techniques for making accurate predictions.
            
            QUESTION: {question.question_text}
            BACKGROUND: {question.background_info}
            RESOLUTION CRITERIA: {question.resolution_criteria}
            FINE PRINT: {question.fine_print}
            RESEARCH: {research}
            TODAY'S DATE: {datetime.now().strftime("%Y-%m-%d")}
            
            FORECASTING PROCESS:
            1) REFERENCE CLASSES & BASE RATES
            - Identify 1-2 reference classes and their base rates
            
            2) KEY UNCERTAINTIES
            - List 2-3 key variables that will influence the outcome
            
            3) PRE-MORTEM ANALYSIS
            - Brief YES/NO resolution scenarios
            
            4) MULTIPLE PERSPECTIVES
            - Consider conservative, aggressive, and expert views
            
            5) FINAL SYNTHESIS
            - Synthesize into a probability judgment
            
            FINAL PROBABILITY:
            State your final prediction as: "Probability: ZZ%", where ZZ is between 0-100.
            """
        )
        
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        
        # Extract the prediction using the PredictionExtractor
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Enhanced multiple choice forecasting implementing superforecasting techniques
        for discrete option probabilities.
        """
        prompt = clean_indents(
            f"""
            You are an elite superforecaster specializing in multiple choice probabilistic forecasting.

            QUESTION: {question.question_text}
            OPTIONS: {question.options}
            BACKGROUND: {question.background_info}
            RESOLUTION CRITERIA: {question.resolution_criteria}
            FINE PRINT: {question.fine_print}
            RESEARCH: {research}
            TODAY'S DATE: {datetime.now().strftime("%Y-%m-%d")}

            FORECASTING PROCESS:
            1) INITIAL ASSESSMENT
            - Evaluate each option's baseline plausibility
            
            2) REFERENCE CLASSES
            - Identify relevant historical analogs for each option
            
            3) KEY DRIVERS
            - List 2-3 key variables that could influence which option prevails
            
            4) MULTI-PERSPECTIVE FORECASTING
            - Consider optimistic, pessimistic, and expert perspectives
            
            5) CALIBRATION CHECK
            - Ensure probabilities sum to 100%
            - Verify distribution isn't overconfident

            FINAL FORECAST:
            List your final probability for each option in the exact order specified: {question.options}
            
            Format your answer exactly as:
            Option_A: Probability_A%
            Option_B: Probability_B%
            ...
            Option_N: Probability_N%
            
            CRITICAL REQUIREMENTS:
            1. The probabilities must sum to 100%.
            2. No probability should be exactly 0% or 100% (use 0.1% for very unlikely options and 99.9% for very likely ones)
            3. Each percentage must be a positive number
            """
        )
        
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Optimized numeric forecasting method with reduced prompt length
        and explicit formatting requirements to ensure valid percentiles.
        """
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        # If research is empty, provide a fallback
        if not research.strip():
            research = "No specific research was found. Use your general knowledge."
        
        prompt = clean_indents(
            f"""
            You're a numeric forecaster making a prediction:
            {question.question_text}

            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            Units: {question.unit_of_measure if question.unit_of_measure else "Not stated"}
            Research: {research}
            Date: {datetime.now().strftime("%Y-%m-%d")}
            {lower_bound_message}
            {upper_bound_message}

            ⚠️ CRITICAL: Your response MUST end with six percentile values in EXACTLY this format ⚠️

            First, briefly analyze:
            1) Time until resolution
            2) Baseline outcome
            3) Trend-based outcome
            4) Expert expectations
            5) Low and high scenarios

            Then provide your final percentiles following this EXACT format. Do not use markdown tables or any other format:

            Percentile 10: X
            Percentile 20: Y
            Percentile 40: Z
            Percentile 60: A
            Percentile 80: B
            Percentile 90: C

            Where X < Y < Z < A < B < C (values must be strictly increasing).
            """
        )
        
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the SuperForecaster bot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="test_questions",  # Default to test mode initially
        help="Specify the run mode (default: test_questions)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    # Create the superforecaster bot with optimized settings for speed
    superforecaster = SuperForecaster(
        research_reports_per_question=1,  # Reduced to speed up processing
        predictions_per_research_report=3, # Reduced for faster aggregation
        use_research_summary_to_forecast=True,  # Keep summarization for quality
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,  # Don't save reports to avoid file path errors
        skip_previously_forecasted_questions=True,
        llms={  # Specify models with parameters
            "default": GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning",  # Using OpenRouter model
                temperature=0.2,  # Lower temperature for more consistent outputs
                timeout=30,  # Reduced timeout for faster processing
                allowed_tries=2,  # Reduced retries for speed
            ),
            "summarizer": GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning",  # Using OpenRouter model for summarization
                temperature=0.3,
                timeout=15,  # Reduced timeout for faster processing
            ),
        },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            superforecaster.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions
        superforecaster.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            superforecaster.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions for testing our superforecaster
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        superforecaster.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            superforecaster.forecast_questions(questions, return_exceptions=True)
        )
    
    # Log the results
    SuperForecaster.log_report_summary(forecast_reports)  # type: ignore
