import google.generativeai as genai
from typing import Dict
from cricket_ai.utils.config import settings
from cricket_ai.utils.logger import get_logger

log = get_logger()

def _mock_explanation(ctx: Dict, conf: float) -> str:
    """
    Rule-based fallback explanation if no API key or LLM fails.
    """
    if conf >= 0.85:
        return "Based on the current match situation, the chasing team has a very high chance of winning because they are performing exceptionally well with a strong run rate and favorable conditions."
    if conf >= 0.60:
        return "Based on the current match situation, the chasing team has a good chance of winning, but the outcome could still shift depending on how they handle the remaining overs."
    if conf < 0.30:
        return "Based on the current match situation, the chasing team has a very low chance of winning and is likely to lose due to challenging circumstances with wickets and required run rate."
    return "The match is finely balanced. The chasing team has a moderate chance of winning, and it could go either way depending on the next few overs."

def explain_with_gemini(match_context: dict, confidence: float) -> str:
    """
    Generate a natural language explanation using Gemini Flash.
    Falls back to a rule-based explanation if no API key or an error occurs.
    """
    if not settings.GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set. Using mock explanation.")
        return _mock_explanation(match_context, confidence)

    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are a cricket commentator explaining a match prediction to fans.
        
        Match Context:
        - Current runs: {match_context.get('total_runs', 'N/A')}
        - Wickets fallen: {match_context.get('wickets', 'N/A')}
        - Target to win: {match_context.get('target', 'N/A')}
        - Balls remaining: {match_context.get('balls_left', 'N/A')}
        - Current run rate: {match_context.get('current_run_rate', 'N/A'):.2f}
        - Required run rate: {match_context.get('required_run_rate', 'N/A'):.2f}
        
        The AI model predicts the chasing team has a {confidence:.1%} chance of winning.
        
        Write a brief, engaging explanation (2-3 sentences) that a cricket fan would understand. Focus on the key factors affecting the prediction.
        """

        response = model.generate_content(prompt)

        if response and response.text:
            return response.text.strip()
        else:
            log.warning("Empty response from Gemini. Falling back to mock.")
            return _mock_explanation(match_context, confidence)

    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return _mock_explanation(match_context, confidence)

def explain(match_context: dict, confidence: float) -> str:
    """
    Main explanation function that uses Gemini if available, otherwise falls back to mock.
    """
    return explain_with_gemini(match_context, confidence)
