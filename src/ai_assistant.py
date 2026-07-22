"""Dataset-grounded multi-turn Gemini business assistant."""

from __future__ import annotations

import json
import os
import pandas as pd


DEFAULT_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
)


def _context(raw_df=None, prediction_df=None, recs_df=None, summary=None):
    ctx = {"business_summary": summary or {}}
    if isinstance(raw_df, pd.DataFrame):
        ctx["dataset"] = {"rows": len(raw_df), "columns": list(raw_df.columns)}
    if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
        cols = [c for c in ["Product_ID", "Product_Name", "Category", "Supplier", "Store_City", "Current_Stock", "Days_To_Expiry", "Predicted_Expiry_Risk", "Predicted_Sales_Velocity", "Predicted_Revenue", "Predicted_Revenue_at_Risk"] if c in prediction_df]
        ranked = prediction_df.sort_values("Predicted_Revenue_at_Risk", ascending=False) if "Predicted_Revenue_at_Risk" in prediction_df else prediction_df
        ctx["priority_products"] = json.loads(ranked[cols].head(30).to_json(orient="records"))
    if isinstance(recs_df, pd.DataFrame) and not recs_df.empty:
        ctx["recommendations"] = json.loads(recs_df.head(30).to_json(orient="records"))
    return ctx


def answer(question, history, raw_df=None, prediction_df=None, recs_df=None, summary=None):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    context = _context(raw_df, prediction_df, recs_df, summary)
    if not key:
        return "AI Assistant is not connected yet. Open the project .env file, add your key after GEMINI_API_KEY=, save it, and restart Streamlit."
    try:
        from google import genai
        client = genai.Client(api_key=key)
        prompt = f"""You are SmartInventory AI, a senior inventory and business decision copilot.
Answer the current user naturally and remember the recent conversation. For company-specific claims use only CONTEXT. Never invent numbers. You may provide clearly labelled general business guidance. Explain what matters, why, priority, business impact and next action. If data is insufficient, name the missing fields. Do not repeat the previous answer.

CONTEXT: {json.dumps(context, default=str)}
RECENT CONVERSATION: {json.dumps(history[-8:], default=str)}
CURRENT QUESTION: {question}
"""
        preferred = os.getenv("GEMINI_MODEL", "").strip()
        candidates = tuple(dict.fromkeys(([preferred] if preferred else []) + list(DEFAULT_MODELS)))
        failures = []
        for model_name in candidates:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    return text
                failures.append(f"{model_name}: empty response")
            except Exception as model_error:
                failures.append(f"{model_name}: {str(model_error)[:120]}")
        raise RuntimeError(" | ".join(failures))
    except Exception as exc:
        detail = str(exc).lower()
        if "api_key" in detail or "api key" in detail or "401" in detail or "403" in detail:
            return "AI authentication failed. Check GEMINI_API_KEY in .env, save the file, and restart the application."
        if "quota" in detail or "429" in detail:
            return "The Gemini quota is temporarily exhausted. Please wait briefly or review the API project's quota, then try again."
        return "The AI service is temporarily unavailable for the configured models. Your inventory pipeline is safe; check GEMINI_MODEL in .env and try again."
