import json
import re
from typing import Any, Optional

def extract_json_block(text: str) -> str:
    """
    Extract the first continuous JSON-like block (starting with { and ending with }).
    This helps skip LLM chatter before or after the JSON.
    """
    # Find the first occurrences of { and the last occurrence of }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def safe_json_loads(raw: str) -> dict:
    """
    Safely load JSON from a string that may contain:
    1. Markdown code blocks (```json ... ```)
    2. Prefix/suffix text
    3. Unescaped control characters (newlines, tabs) inside strings
    """
    if not raw:
        return {}

    # 1. Strip markdown syntax if present
    cleaned = re.sub(r"```json|```", "", raw).strip()
    
    # 2. Extract only the JSON part
    json_str = extract_json_block(cleaned)
    
    try:
        # 3. Parse with strict=False to allow unescaped control characters
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        # 4. If it fails, try some common cleaning
        try:
            # Replace actual newlines within strings with \n (fragile but sometimes helps)
            # This is a bit risky, so we only try it as a last resort
            # Actually, strict=False should have handled most of this.
            # Let's just log and raise for now to debug if strict=False isn't enough.
            raise e
        except Exception:
            print(f"Failed to parse JSON: {raw[:200]}...")
            return {}
