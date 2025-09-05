import google.generativeai as genai
from cricket_ml.utils.config import settings
from cricket_ml.utils.logger import get_logger

log = get_logger()

def _mock_explanation(match_context: dict, confidence: float) -> str:
    """Fallback explanation if no API key or LLM fails."""
    if confidence >= 0.85:
        return "The chasing team is in a strong position and likely to win based on current runs, wickets, and balls left."
    elif confidence >= 0.60:
        return "The chasing team has a good chance to win, but the match outcome could still change depending on the next few overs."
    elif confidence < 0.30:
        return "The chasing team is in a difficult position and likely to lose given the current match situation."
    else:
        return "The match is evenly balanced. Both teams have a fair chance depending on upcoming plays."

def explain_with_gemini(match_context: dict, confidence: float) -> str:
    """
    Generate concise natural-language explanation using Gemini Flash.
    Falls back to mock explanation if API key is missing or error occurs.
    """
    if not settings.GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set. Using mock explanation.")
        return _mock_explanation(match_context, confidence)

    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
    You are a friendly cricket commentator. Explain the match prediction
    to casual cricket fans in simple, human-readable language. Keep it short
    (1-2 sentences) and clear.

    Match Details:
    - Total runs scored: {match_context.get('total_runs', 'N/A')}
    - Wickets fallen: {match_context.get('wickets', 'N/A')}
    - Target runs: {match_context.get('target', 'N/A')}
    - Balls remaining: {match_context.get('balls_left', 'N/A')}
    - Current run rate: {match_context.get('current_run_rate', 'N/A')}
    - Required run rate: {match_context.get('required_run_rate', 'N/A')}

    AI Prediction:
    - Probability of batting side winning: {confidence:.1%}

    Instruction: 
    Write a short, casual, easy-to-read explanation for fans, e.g. "The batting side has 8 wickets left and only 36 runs needed from 59 balls. They are in full control and likely to win comfortably."
    """
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            log.warning("Empty response from Gemini. Using fallback.")
            return _mock_explanation(match_context, confidence)

    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return _mock_explanation(match_context, confidence)

def explain(match_context: dict, confidence: float) -> str:
    """Main function for /explain endpoint."""
    return explain_with_gemini(match_context, confidence)
