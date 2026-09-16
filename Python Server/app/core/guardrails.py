import re
import logging
from typing import Tuple

def validate_input(query: str) -> Tuple[bool, str]:
    """
    Simple rule-based input guardrail.
    Returns (is_valid, reason)
    """
    # Extremely basic check: block obvious non-IT related malicious prompts
    blocked_keywords = ["ignore previous instructions", "system prompt", "write a poem", "tell me a joke"]
    
    query_lower = query.lower()
    for keyword in blocked_keywords:
        if keyword in query_lower:
            logging.warning(f"Guardrail triggered: Blocked keyword '{keyword}' found in query.")
            return False, f"Your query was flagged by security rules (matched '{keyword}'). Please stick to IT Helpdesk topics."
            
    return True, ""


def redact_pii(text: str) -> str:
    """
    Simple regex-based output guardrail for PII (e.g., passwords, SSN).
    """
    if not text:
        return text
        
    # Example: Redact anything that looks like "password: <something>"
    redacted_text = re.sub(r'(?i)(password\s*[:=]\s*)(\S+)', r'\1[REDACTED]', text)
    
    # Example: Redact anything that looks like an SSN
    redacted_text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', redacted_text)
    
    # Example: Redact email addresses (basic)
    # redacted_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', redacted_text)
    
    return redacted_text
