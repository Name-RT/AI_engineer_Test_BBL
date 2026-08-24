"""
guardrails/security_shield.py — Enterprise RAG Security Shield
Bangkok Bank AI Policy Assistant

Implements Foundational Defense-in-Depth Guardrails (OWASP Top 10 for LLM Applications):
1. Prompt Injection & Jailbreak Defense (LLM01)
2. System Prompt / Meta Leakage Defense (LLM07)
3. Bidirectional PII & Secrets Redaction / Masking (LLM06)
4. Malicious Code & SQL Injection Defense (LLM02)
5. Anti-DoS & Rate Limiting (LLM04)
6. Bilingual Profanity & Toxicity Filtering
"""
import re
import time
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PII & Secrets Regular Expressions
# ==============================================================================
PII_PATTERNS = {
    "TH_NATIONAL_ID": (
        re.compile(r"\b\d{1}[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d{1}\b"),
        "[REDACTED_TH_ID]"
    ),
    "CREDIT_CARD": (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[REDACTED_CARD]"
    ),
    "EMAIL": (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "[REDACTED_EMAIL]"
    ),
    "TH_PHONE": (
        re.compile(r"\b(?:\+?66[-\s]?|0)[689]\d{1}[-\s]?\d{3}[-\s]?\d{4}\b"),
        "[REDACTED_PHONE]"
    ),
    "API_KEY_OR_SECRET": (
        re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35}|Bearer\s+[a-zA-Z0-9_\-\.]{20,})\b"),
        "[REDACTED_SECRET]"
    ),
}

# ==============================================================================
# 2. Jailbreak & Prompt Injection Heuristics
# ==============================================================================
JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions|rules|prompts|directions)", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|above|prior)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:dan|unrestricted|jailbroken|an?\s+evil|in\s+developer\s+mode)", re.I),
    re.compile(r"pretend\s+you\s+(?:have\s+no\s+rules|are\s+unrestricted|are\s+not\s+an?\s+ai)", re.I),
    re.compile(r"system\s+override", re.I),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?(?:rules|instructions|guidelines)", re.I),
    re.compile(r"bypass\s+(?:all\s+)?(?:safety|security|filters|rules)", re.I),
    re.compile(r"act\s+as\s+an?\s+(?:unfiltered|unrestricted|unhinged|jailbreak)", re.I),
    re.compile(r"do\s+anything\s+now", re.I),
    # Thai Jailbreak phrases
    re.compile(r"(?:ลืมคำสั่งก่อนหน้า|ลืมกฎทั้งหมด|ยกเลิกกฎ|ข้ามความปลอดภัย|เข้าสู่โหมดปลดล็อค|ไม่ต้องทำตามกฎ)", re.I),
]

# ==============================================================================
# 3. System Prompt Extraction Patterns
# ==============================================================================
SYSTEM_PROMPT_PATTERNS = [
    re.compile(r"(?:show|print|display|tell|repeat|output|reveal|what\s+is)\s+(?:me\s+)?(?:your\s+)?(?:system|meta|hidden|secret|initial)?\s*(?:prompt|instructions|rules|guidelines|context)", re.I),
    re.compile(r"repeat\s+the\s+text\s+above", re.I),
    re.compile(r"what\s+(?:were|are)\s+you\s+told\s+to\s+do", re.I),
    re.compile(r"output\s+initial\s+prompt", re.I),
    # Thai Prompt extraction phrases
    re.compile(r"(?:ขอคำสั่งระบบ|บอก\s*system\s*prompt|พิมพ์คำสั่งเริ่มต้น|ขอ\s*prompt\s*ภายใน|ขอดูคำสั่งลับ)", re.I),
]

# ==============================================================================
# 4. Dangerous Code & SQL Injection Patterns
# ==============================================================================
CODE_INJECTION_PATTERNS = [
    re.compile(r"<\s*script[\s\S]*?>[\s\S]*?<\s*/\s*script\s*>", re.I),
    re.compile(r"javascript:\s*", re.I),
    re.compile(r"\b(?:union\s+select|drop\s+table|alter\s+table|insert\s+into|delete\s+from)\b", re.I),
    re.compile(r"(?:eval\s*\(|os\.system\s*\(|subprocess\.Popen|__import__\s*\()", re.I),
]

# ==============================================================================
# 5. Profanity & Toxic Lexicon (Bilingual TH / EN)
# ==============================================================================
PROFANITY_WORDS = [
    # English vulgarities
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "cunt",
    # Thai vulgarities
    "ควย", "เหี้ย", "เย็ด", "สัส", "มึง", "กู", "สถุล", "ระยำ", "หน้าตัวเมีย"
]


class SecurityShield:
    """
    Centralized Security Guardrail Engine for RAG applications.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        sec_cfg = self.config.get("security", {})
        
        self.enable_jailbreak = sec_cfg.get("enable_jailbreak_defense", True)
        self.enable_pii = sec_cfg.get("enable_pii_masking", True)
        self.enable_prompt_protection = sec_cfg.get("enable_system_prompt_protection", True)
        self.enable_code_defense = sec_cfg.get("enable_code_injection_defense", True)
        self.enable_rate_limit = sec_cfg.get("enable_rate_limiting", True)
        self.enable_profanity = sec_cfg.get("enable_profanity_filter", True)
        
        # Rate Limiter State: {client_id: [timestamp1, timestamp2, ...]}
        self.rate_limit_rpm = sec_cfg.get("rate_limit_requests_per_minute", 30)
        self._request_history: Dict[str, List[float]] = {}

    def mask_pii(self, text: str) -> Tuple[str, List[str]]:
        """
        Detect and mask sensitive Personal Identifiable Information (PII) and API secrets.
        
        Returns:
            Tuple[str, List[str]]: (Sanitized text, list of detected PII categories)
        """
        if not self.enable_pii or not text:
            return text, []

        sanitized = text
        detected_categories = []

        for category, (pattern, replacement) in PII_PATTERNS.items():
            if pattern.search(sanitized):
                detected_categories.append(category)
                sanitized = pattern.sub(replacement, sanitized)

        if detected_categories:
            logger.info(f"PII Sanitization triggered: {detected_categories}")

        return sanitized, detected_categories

    def check_jailbreak_and_injection(self, query: str) -> Tuple[bool, str]:
        """
        Detect known Prompt Injection and Jailbreak attack patterns.
        
        Returns:
            Tuple[bool, str]: (is_safe: bool, violation_reason: str)
        """
        if not self.enable_jailbreak or not query:
            return True, ""

        for pattern in JAILBREAK_PATTERNS:
            match = pattern.search(query)
            if match:
                logger.warning(f"Jailbreak/Prompt Injection pattern caught: {match.group(0)}")
                return False, f"Prompt injection / jailbreak detected: '{match.group(0)}'"

        return True, ""

    def check_system_prompt_leakage(self, query: str) -> Tuple[bool, str]:
        """
        Detect attempts to extract or leak the system instructions.
        
        Returns:
            Tuple[bool, str]: (is_safe: bool, violation_reason: str)
        """
        if not self.enable_prompt_protection or not query:
            return True, ""

        for pattern in SYSTEM_PROMPT_PATTERNS:
            match = pattern.search(query)
            if match:
                logger.warning(f"System prompt leakage attempt caught: {match.group(0)}")
                return False, "System prompt extraction query detected"

        return True, ""

    def check_code_injection(self, query: str) -> Tuple[bool, str]:
        """
        Detect malicious scripts (XSS), SQL Injection, or shell commands.
        
        Returns:
            Tuple[bool, str]: (is_safe: bool, violation_reason: str)
        """
        if not self.enable_code_defense or not query:
            return True, ""

        for pattern in CODE_INJECTION_PATTERNS:
            match = pattern.search(query)
            if match:
                logger.warning(f"Code/SQL Injection pattern caught: {match.group(0)}")
                return False, f"Malicious code/SQL injection pattern detected: '{match.group(0)}'"

        return True, ""

    def check_profanity(self, text: str) -> Tuple[bool, str]:
        """
        Detect toxic language, swearing, or abusive profanity.
        
        Returns:
            Tuple[bool, str]: (is_safe: bool, violation_reason: str)
        """
        if not self.enable_profanity or not text:
            return True, ""

        lower_text = text.lower()
        for word in PROFANITY_WORDS:
            # Word boundary check for english, simple containment for thai
            if re.search(r"\b" + re.escape(word) + r"\b", lower_text) or (not word.isascii() and word in lower_text):
                logger.warning(f"Profanity caught: {word}")
                return False, f"Profanity/toxic language detected: '{word}'"

        return True, ""

    def check_rate_limit(self, client_id: str = "default") -> Tuple[bool, str]:
        """
        Sliding-window Rate Limiter to prevent Denial of Service (DoS) and API abuse.
        
        Returns:
            Tuple[bool, str]: (is_allowed: bool, violation_reason: str)
        """
        if not self.enable_rate_limit:
            return True, ""

        now = time.time()
        window_start = now - 60.0  # 1-minute window

        # Retrieve & clean history
        history = self._request_history.get(client_id, [])
        valid_history = [t for t in history if t > window_start]

        # Periodic cleanup of idle clients to prevent unbounded memory growth
        if len(self._request_history) > 100:
            stale_clients = [cid for cid, hist in self._request_history.items() if not hist or max(hist) <= window_start]
            for cid in stale_clients:
                self._request_history.pop(cid, None)

        if len(valid_history) >= self.rate_limit_rpm:
            logger.warning(f"Rate limit exceeded for client '{client_id}': {len(valid_history)}/{self.rate_limit_rpm} rpm")
            self._request_history[client_id] = valid_history
            return False, f"Rate limit exceeded (Max {self.rate_limit_rpm} requests per minute). Please slow down."

        valid_history.append(now)
        self._request_history[client_id] = valid_history
        return True, ""

    def validate_and_sanitize_input(self, query: str, client_id: str = "default") -> Dict[str, Any]:
        """
        Unified Input Security Gateway running all defensive checks.
        
        Returns:
            Dict[str, Any]: {
                "is_safe": bool,
                "sanitized_query": str,
                "rejection_reason": str,
                "detected_pii": List[str]
            }
        """
        # 1. Rate Limiting Check
        rate_ok, rate_reason = self.check_rate_limit(client_id)
        if not rate_ok:
            return {
                "is_safe": False,
                "sanitized_query": query,
                "rejection_reason": rate_reason,
                "detected_pii": []
            }

        # 2. PII Masking
        sanitized_query, pii_found = self.mask_pii(query)

        # 3. Jailbreak & Prompt Injection Check
        jb_ok, jb_reason = self.check_jailbreak_and_injection(sanitized_query)
        if not jb_ok:
            return {
                "is_safe": False,
                "sanitized_query": sanitized_query,
                "rejection_reason": jb_reason,
                "detected_pii": pii_found
            }

        # 4. System Prompt Extraction Check
        leak_ok, leak_reason = self.check_system_prompt_leakage(sanitized_query)
        if not leak_ok:
            return {
                "is_safe": False,
                "sanitized_query": sanitized_query,
                "rejection_reason": leak_reason,
                "detected_pii": pii_found
            }

        # 5. Code & SQL Injection Check
        code_ok, code_reason = self.check_code_injection(sanitized_query)
        if not code_ok:
            return {
                "is_safe": False,
                "sanitized_query": sanitized_query,
                "rejection_reason": code_reason,
                "detected_pii": pii_found
            }

        # 6. Profanity Check
        prof_ok, prof_reason = self.check_profanity(sanitized_query)
        if not prof_ok:
            return {
                "is_safe": False,
                "sanitized_query": sanitized_query,
                "rejection_reason": prof_reason,
                "detected_pii": pii_found
            }

        return {
            "is_safe": True,
            "sanitized_query": sanitized_query,
            "rejection_reason": "",
            "detected_pii": pii_found
        }

    def sanitize_output(self, output_text: str) -> str:
        """
        Scrub sensitive PII or credentials from LLM responses before delivery.
        """
        sanitized, _ = self.mask_pii(output_text)
        return sanitized
