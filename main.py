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
    Enhanced forecasting bot implementing superforecasting principles:
    - Aggregates multiple diverse research sources for more robust information
    - Uses reference class forecasting to ground predictions in base rates
    - Implements Tetlock's superforecasting principles: probabilistic thinking, belief updating, etc.
    - Multiple prediction runs for higher accuracy and confidence intervals

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    General flow:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions using aggregation techniques from superforecasting
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    If needed, a more sophisticated rate limiter is available:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially
    await self.rate_limiter.wait_till_able_to_acquire_resources(1)
    ```
    """

    _max_concurrent_questions = 3  # Increased for more parallel processing
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Enhanced research function that tries to combine multiple sources 
        when available for a more comprehensive analysis.
        """
        async with self._concurrency_limiter:
            research_sources = []
            
            # Try to get research from each available provider
            if os.getenv("EXA_API_KEY"):
                try:
                    exa_research = await self._call_exa_smart_searcher(question.question_text)
                    if exa_research:
                        research_sources.append(f"--- EXA SEARCH RESULTS ---\n{exa_research}")
                except Exception as e:
                    logger.warning(f"Error using Exa research for {question.page_url}: {str(e)}")
            
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                try:
                    asknews_research = await AskNewsSearcher().get_formatted_news_async(question.question_text)
                    if asknews_research:
                        research_sources.append(f"--- ASKNEWS SEARCH RESULTS ---\n{asknews_research}")
                except Exception as e:
                    logger.warning(f"Error using AskNews research for {question.page_url}: {str(e)}")
            
            if os.getenv("PERPLEXITY_API_KEY"):
                try:
                    perplexity_research = await self._call_perplexity(question.question_text)
                    if perplexity_research:
                        research_sources.append(f"--- PERPLEXITY SEARCH RESULTS ---\n{perplexity_research}")
                except Exception as e:
                    logger.warning(f"Error using Perplexity research for {question.page_url}: {str(e)}")
            
            elif os.getenv("OPENROUTER_API_KEY") and not research_sources:
                try:
                    openrouter_research = await self._call_perplexity(question.question_text, use_open_router=True)
                    if openrouter_research:
                        research_sources.append(f"--- OPENROUTER SEARCH RESULTS ---\n{openrouter_research}")
                except Exception as e:
                    logger.warning(f"Error using OpenRouter research for {question.page_url}: {str(e)}")
            
            # Combine all research sources
            if research_sources:
                combined_research = "\n\n".join(research_sources)
                
                # If we have multiple sources, add a summarization step
                if len(research_sources) > 1:
                    try:
                        summary = await self._summarize_multi_source_research(combined_research, question.question_text)
                        combined_research = f"--- RESEARCH SUMMARY ---\n{summary}\n\n--- DETAILED SOURCES ---\n{combined_research}"
                    except Exception as e:
                        logger.warning(f"Error summarizing multi-source research: {str(e)}")
                
                logger.info(f"Found Research for URL {question.page_url}")
                return combined_research
            else:
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
        Summarizes research from multiple sources to create a unified view, highlighting 
        agreements and contradictions.
        """
        summarization_prompt = clean_indents(
            f"""
            You are a research analyst assisting a superforecaster.
            
            QUESTION TO FORECAST:
            {question}
            
            You have been given research from multiple sources, and your job is to synthesize 
            this information into a cohesive summary. Focus on:
            
            1. Key facts and data points that appear across multiple sources (high confidence)
            2. Important points mentioned by only one source (require verification)
            3. Any contradictions between sources (highlight uncertainty)
            4. Relevant base rates and reference classes for this type of question
            5. Current trends and how they might evolve
            6. Identify what information is still missing that would be valuable
            
            FORMAT YOUR RESPONSE:
            - Summary of Key Facts (75-150 words)
            - Critical Data Points (bullet points of numbers, statistics, dates)
            - Identified Reference Classes & Base Rates
            - Areas of Uncertainty
            - Current Trajectory
            
            RESEARCH FROM MULTIPLE SOURCES:
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
            
            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}
            
            FINE PRINT:
            {question.fine_print}

            RESEARCH:
            {research}

            TODAY'S DATE:
            {datetime.now().strftime("%Y-%m-%d")}
            
            FORECASTING PROCESS:
            Follow this structured superforecasting approach carefully:
            
            1) REFERENCE CLASSES & BASE RATES
            - Identify at least 2-3 relevant reference classes for this question
            - Research and state the base rates for each reference class 
            - Assess which reference class is most applicable and why
            
            2) OUTSIDE VIEW VS INSIDE VIEW
            - Outside view: What does history suggest about questions like this?
            - Inside view: What specific factors make this situation unique?
            - How should we weigh outside vs inside view for this question?
            
            3) KEY UNCERTAINTIES & VARIABLES
            - List 3-5 key variables that will influence the outcome
            - For each variable, estimate its impact on the probability
            
            4) PRE-MORTEM ANALYSIS
            - Imagine it is resolution date and the answer is YES. Why did this happen?
            - Imagine it is resolution date and the answer is NO. Why did this happen?
            
            5) FERMI DECOMPOSITION
            - Break down this question into sub-components if applicable
            - Calculate a probability based on the decomposition
            
            6) TIME HORIZON CONSIDERATIONS
            - Analyze how the time until resolution affects your forecast
            - Consider scenarios where events accelerate or decelerate
            
            7) MULTIPLE PERSPECTIVES
            - Perspective 1: What would a conservative forecaster predict?
            - Perspective 2: What would an aggressive forecaster predict?
            - Perspective 3: What would a domain expert predict?
            
            8) FINAL SYNTHESIS
            - Synthesize the above analyses into a probability judgment
            - Explain which factors were most influential in your forecast
            
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

            QUESTION:
            {question.question_text}

            OPTIONS:
            {question.options}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            RESEARCH:
            {research}

            TODAY'S DATE:
            {datetime.now().strftime("%Y-%m-%d")}

            FORECASTING PROCESS:
            Follow this structured superforecasting approach for multiple choice questions:

            1) INITIAL ASSESSMENT
            - For each option, identify its initial plausibility
            - Consider the status quo and what would happen if nothing changed
            - Identify which option(s) are considered the most likely by experts/markets

            2) REFERENCE CLASSES
            - For each option, identify relevant reference classes and base rates
            - Compare the current situation to historical analogs
            - Estimate the frequency with which similar options prevailed in comparable situations

            3) OPTION-SPECIFIC ANALYSIS
            - For each option, analyze the specific conditions needed for it to be the outcome
            - Estimate the likelihood of those conditions occurring
            - Consider unique factors that make each option more or less likely than historical base rates

            4) KEY DRIVERS & UNCERTAINTIES
            - Identify the key variables that could influence which option prevails
            - For each variable, assess how likely it is to move in a direction favorable to each option
            - Consider contingencies and dependencies between variables

            5) SCENARIO MAPPING
            - Map different possible future scenarios to each option
            - Consider specific paths and timelines that lead to each option
            - Identify critical junctures and decision points

            6) CROSS-IMPACT ANALYSIS
            - Analyze how the options interact with one another
            - Consider whether some options are mutually exclusive or complementary
            - Account for the possibility that the "correct" option might be a combination of listed options

            7) MULTI-PERSPECTIVE FORECASTING
            - Adopt different perspectives (optimistic, pessimistic, status quo, domain expert, etc.)
            - Generate probability distributions from each perspective
            - Reconcile these perspectives into a single coherent distribution

            8) CALIBRATION & COHERENCE CHECK
            - Ensure probabilities sum to 100%
            - Leave appropriate probability mass on "unlikely" options to account for surprises
            - Check that your distribution isn't overconfident or underconfident
            - Consider whether your relative confidences across options are justified

            FINAL FORECAST:
            List your final probability for each option in the exact order specified: {question.options}
            
            Format your answer exactly as:
            Option_A: Probability_A%
            Option_B: Probability_B%
            ...
            Option_N: Probability_N%
            
            The probabilities must sum to 100%.
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
        Enhanced numeric forecasting method implementing superforecasting techniques
        for continuous distributions and calibration.
        """
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        prompt = clean_indents(
            f"""
            You are a world-class superforecaster specializing in quantitative predictions using techniques from statistics, decision science, and cognitive psychology.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            UNITS FOR ANSWER: 
            {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            RESEARCH:
            {research}

            TODAY'S DATE:
            {datetime.now().strftime("%Y-%m-%d")}

            BOUNDS:
            {lower_bound_message}
            {upper_bound_message}

            FORECASTING PROCESS:
            Follow this structured process for making a precise quantitative forecast:

            1) KEY METRICS & HISTORICAL DATA
            - Identify key metrics related to this question
            - Analyze historical trends and growth/decline rates
            - Calculate relevant summary statistics (mean, median, growth rates)

            2) COMPARABLE REFERENCE POINTS
            - Identify comparable situations, entities, or historical periods 
            - Extract numeric values from these comparables
            - Calculate typical ranges and distributions

            3) MODEL BUILDING
            - Develop a simple model of the factors influencing this number
            - Weight each factor by importance
            - Consider how factors interact (multiplicative vs. additive effects)

            4) SCENARIO ANALYSIS
            - Baseline scenario: What happens if current trends continue?
            - Pessimistic scenario: What realistic factors could drive the number lower?
            - Optimistic scenario: What realistic factors could drive the number higher?
            - Wild card scenario: What low-probability events could dramatically change the outcome?

            5) MONTE CARLO SIMULATION REASONING
            - Consider the distribution shape (normal, log-normal, power law, etc.)
            - Identify key uncertainties and their distributions
            - Reason through how these uncertainties would combine

            6) FORECAST DISTRIBUTION
            - Generate a full probability distribution, not just a point estimate
            - Identify the median, mean, and various percentiles (10, 20, 40, 60, 80, 90)
            - Check that these values are coherent (e.g., 80th percentile > 60th percentile)
            - Ensure your distribution accounts for tail risks and unknown unknowns
            - Double-check that your distribution respects any hard bounds specified

            7) CALIBRATION CHECK
            - How often have past forecasts in this domain been too high or too low?
            - Do you need to widen your confidence intervals to avoid overconfidence?
            - Would other forecasters produce similar or different distributions?

            FORMATTING INSTRUCTIONS:
            - Use the exact units requested (e.g., whether a number should be 1,000,000 or 1 million)
            - Never use scientific notation
            - Use comma separators for thousands (e.g., 1,234,567)
            - Always arrange percentiles from smaller to larger numbers

            FINAL FORECAST:
            Provide your distribution as:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
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

    # Create the superforecaster bot with configuration using OpenRouter models
    superforecaster = SuperForecaster(
        research_reports_per_question=2,  # Increased from 1 to get more diverse research
        predictions_per_research_report=5, # Increased for better aggregation
        use_research_summary_to_forecast=True,  # Enable summarization of research
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,  # Don't save reports to avoid file path errors
        skip_previously_forecasted_questions=True,
        llms={  # Specify models with parameters
            "default": GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning",  # Using OpenRouter model
                temperature=0.2,  # Lower temperature for more consistent outputs
                timeout=60,  # Increased timeout for more thorough analysis
                allowed_tries=3,  # More retries
            ),
            "summarizer": GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning",  # Using OpenRouter model for summarization
                temperature=0.3,
                timeout=30,
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
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
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
