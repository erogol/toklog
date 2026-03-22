"""BM25+ text classifier for use case classification of log entries."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_CATEGORIES_FILE = Path.home() / ".toklog" / "categories.json"

DEFAULT_CATEGORIES: List[Dict[str, str]] = [
    {"name": "code_generation", "description": "Writing, fixing, refactoring, or generating code and programming implementations"},
    {"name": "summarization", "description": "Summarizing, condensing, or extracting key points from documents and text"},
    {"name": "chat", "description": "Casual conversation, opinions, recommendations, and general questions"},
    {"name": "research", "description": "Researching topics, comparing technologies, reviewing literature, and analyzing trends"},
    {"name": "data_extraction", "description": "Extracting, parsing, scraping, or converting structured data from various sources"},
    {"name": "tool_use", "description": "Using external tools, running commands, sending messages, or triggering actions"},
]

# ---------------------------------------------------------------------------
# Persistence (unchanged)
# ---------------------------------------------------------------------------


def load_categories() -> List[Dict[str, str]]:
    """Load categories from disk. Returns empty list if file missing."""
    try:
        with open(_CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_categories(categories: List[Dict[str, str]]) -> None:
    """Save categories to disk, creating parent directory if needed."""
    _CATEGORIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)


# ---------------------------------------------------------------------------
# Tokenizer & stemmer
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if t]


def simple_stem(word: str) -> str:
    """Simple suffix-stripping stemmer — no external deps."""
    for suffix in ["tion", "ment", "ness", "ing", "ally", "ous", "ive",
                    "ful", "less", "able", "ible", "ly", "ed", "er", "es", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "code_generation": [
        "write", "implement", "create", "build", "code", "function", "class", "method",
        "fix", "bug", "refactor", "generate", "script", "program", "algorithm", "api",
        "endpoint", "test", "unit", "module", "library", "import", "syntax", "compile",
        "debug", "deploy", "dockerfile", "regex", "parse", "middleware",
        "validator", "authenticate", "handler", "component", "react", "migration",
        "decorator", "iterator", "pooling", "crud", "pagination", "typeahead", "sorting",
        "cache", "lru", "webhook", "stripe", "argparse", "cli", "docker", "kubernetes",
        "grpc", "protobuf", "graphql", "schema", "mutation", "resolver", "redux",
        "webpack", "eslint", "prettier", "typescript", "rust", "java", "python", "sql",
        "benchmark", "microservice", "architecture", "hook", "async", "thread", "socket",
        "cors", "jwt", "oauth", "terraform", "makefile", "shell",
        "ssl", "certificate", "kubernetes", "k8s", "nginx", "apache", "dns", "firewall",
        # tool-name signals
        "bash", "execute_code", "run_code", "code_interpreter", "terminal", "repl",
        "eval", "lint", "format_code", "write_file", "edit_file", "read_file", "patch",
    ],
    "summarization": [
        "summarize", "summary", "condense", "shorten", "brief", "overview", "key",
        "points", "takeaways", "highlights", "gist", "tldr", "recap", "digest",
        "distill", "boil", "essentials", "conclusions", "findings", "bullet",
        "executive", "cliff", "notes", "abstract", "synthesis",
        "main", "compress", "long", "document", "article", "report", "meeting",
        "discussion", "thread", "email", "conversation", "chapter", "paper",
        "feedback", "verbose", "paragraph",
        # tool-name signals
        "read_document", "extract_text", "ocr", "pdf", "parse_document",
    ],
    "chat": [
        "hey", "hello", "hi", "joke", "opinion", "think", "feel", "recommend",
        "suggest", "favorite", "best", "worst", "decide", "help",
        "bored", "fun", "interesting", "explain", "like", "meaning", "life",
        "friend", "name", "idea", "advice", "chat",
        "difference", "between", "about", "know", "understand", "curious", "wonder", "thoughts",
        "question", "mean", "thank",
    ],
    "research": [
        "research", "compare", "comparison", "study", "paper", "papers", "survey",
        "literature", "benchmark", "evaluate", "analysis", "analyze", "trend",
        "state", "art", "advances", "investigate", "review", "tradeoff", "tradeoffs",
        "approach", "technique", "method", "pros", "cons", "latest", "current",
        "discover", "explore", "learn", "which", "better", "alternative",
        "option", "practice", "framework", "technology", "ecosystem", "landscape",
        "assessment", "insight",
        # tool-name signals
        "web_search", "arxiv", "browse", "fetch_url", "wikipedia", "lookup",
        "query", "retrieve", "google",
    ],
    "data_extraction": [
        "extract", "parse", "pull", "scrape", "convert", "get", "grab", "data",
        "csv", "json", "xml", "html", "table", "column", "row", "field",
        "structured", "unstructured", "metadata", "entity", "coordinate", "format",
        "number", "price", "date", "address", "content", "information",
        "record", "entry", "list", "value", "pattern", "match", "clean",
        "normalize", "transform",
    ],
    "tool_use": [
        "search", "send", "email", "calendar", "deploy", "run",
        "execute", "trigger", "post", "fetch", "upload", "download", "book",
        "check", "look", "update", "query", "database", "pipeline",
        "slack", "jira", "ticket", "stock",
        "call", "invoke", "automate", "workflow", "notify", "alert", "monitor",
        "log", "track", "integration", "message", "notification",
        "remind", "reminder", "open",
        "rotate", "restart", "drain", "rollback", "provision", "scale",
        "sync", "credential", "credentials", "health", "verify", "roll",
        "back", "balancer", "backup", "maintenance",
        "instances", "aws", "container", "registry",
        # tool-name signals
        "send_message", "post_tweet", "telegram", "webhook", "api_call",
        "http", "get_request", "schedule_event", "create_event",
    ],
    "agent_planning": [
        "plan", "decide", "next step", "which tool", "strategy", "approach",
        "figure out", "workflow", "orchestrate", "coordinate", "multi-step",
        "proceed", "goal", "breakdown", "sequence", "prioritize", "delegate",
        "subtask", "pipeline", "chain", "route", "dispatch", "schedule",
        "determine", "evaluate options", "pick", "choose", "best way",
        "how should", "what should", "should i", "need to", "order of",
        "depends on", "first then", "step by step",
    ],
    "scheduled_job": [
        "digest", "briefing", "morning", "daily", "cron", "scheduled", "automated",
        "newsletter", "report", "monitor", "alert", "weekly", "recurring",
        "startup ideas", "trending", "roundup", "summary", "update",
        "consolidate", "memory", "recall", "store memory", "session history",
        "long-term memory", "context window", "save fact", "remember",
        "job", "task", "run at", "fire", "trigger",
    ],
}

HIGH_SIGNAL: Dict[str, List[str]] = {
    "code_generation": [
        "implement", "debug", "refactor", "function", "class", "code", "write",
        "build", "create", "script", "program", "algorithm", "fix", "bug",
        "dockerfile", "middleware", "decorator", "handler", "component",
        "migration", "endpoint", "api", "module", "library", "test",
    ],
    "summarization": [
        "summarize", "summary", "condense", "tldr", "recap", "digest",
        "distill", "shorten", "brief", "overview", "gist", "highlights",
        "takeaways", "conclusions", "bullet", "executive",
    ],
    "chat": [
        "joke", "opinion", "think", "recommend", "favorite", "fun",
        "bored", "interesting", "hey", "hello", "hi", "chat", "feel",
        "tell", "explain", "advice", "suggest",
    ],
    "research": [
        "compare", "research", "survey", "investigate", "evaluate", "analyze",
        "literature", "benchmark", "tradeoff", "tradeoffs", "pros", "cons",
        "advances", "state", "art", "review", "trend", "latest",
    ],
    "data_extraction": [
        "extract", "parse", "scrape", "csv", "json", "xml", "html",
        "pull", "grab", "structured", "metadata", "table",
    ],
    "tool_use": [
        "send", "email", "calendar", "schedule", "deploy", "run",
        "execute", "trigger", "fetch", "upload", "slack", "jira",
        "cron", "pipeline", "book", "stock", "dns",
    ],
}

FIRST_VERB_BONUS: Dict[str, List[str]] = {
    "code_generation": [
        "write", "implement", "create", "build", "fix", "refactor",
        "generate", "code", "add", "make", "convert", "port", "design",
        "rewrite",
    ],
    "summarization": [
        "summarize", "condense", "shorten", "recap", "distill", "compress",
        "reduce", "digest", "boil", "outline",
    ],
    "data_extraction": [
        "extract", "parse", "pull", "scrape", "get", "grab",
    ],
    "tool_use": [
        "search", "send", "deploy", "run", "execute", "trigger", "post",
        "fetch", "upload", "book", "schedule", "open", "look", "query", "update",
    ],
    "research": [
        "compare", "evaluate", "analyze", "investigate", "survey",
        "study", "assess", "find",
    ],
    "chat": [
        "tell", "explain", "recommend", "hey", "hello", "hi",
    ],
    "agent_planning": [
        "plan", "decide", "orchestrate", "prioritize",
    ],
    "scheduled_job": [
        "consolidate", "monitor", "schedule",
    ],
}

NEGATIVE_KEYWORDS: Dict[str, List[str]] = {
    "data_extraction": ["write", "implement", "build", "create", "fix", "refactor", "generate"],
    "tool_use": ["write", "implement", "build", "create", "fix", "refactor"],
    "research": ["write", "implement", "build", "create", "fix"],
    "summarization": ["write", "implement", "build", "create", "fix"],
    "chat": ["implement", "refactor", "debug", "compile", "deploy"],
}

# ---------------------------------------------------------------------------
# Pattern bonuses — logically-justified signals only
# ---------------------------------------------------------------------------

PATTERN_BONUSES: List[Tuple[re.Pattern, str, float]] = [
    # Research: URL and citation signals in response output
    (re.compile(r"\bhttps?://"), "research", 2.0),
    (re.compile(r"\bcited?\b", re.I), "research", 2.0),
    (re.compile(r"\baccording to\b", re.I), "research", 2.0),

    # Research: comparison and investigation signals
    (re.compile(r"\b(compare|vs\.?|versus)\b", re.I), "research", 3.5),
    (re.compile(r"\b(best\s+practices?|trade.?offs?|state\s+of)\b", re.I), "research", 3.0),
    (re.compile(r"\bwhat\s+(are|is)\s+the\s+(best|latest|current|emerging|recommended)\b", re.I), "research", 3.5),
    (re.compile(r"\bhow\s+does?\s+\w+\s+compare\b", re.I), "research", 3.5),
    (re.compile(r"\bwhat\s+(does\s+)?(the\s+|recent\s+)?(research|literature|evidence)\s+(say|suggest|show)\b", re.I), "research", 6.0),
    (re.compile(r"\bhow\s+(effective|efficient|reliable|practical)\s+(are|is)\b", re.I), "research", 4.0),
    (re.compile(r"\bwhat\s+(are|is)\s+(the\s+)?(key|main|open|academic)\s+(differences?|implications?|findings?|directions?|challenges?|approaches?)\b", re.I), "research", 7.0),
    (re.compile(r"\bexplain\s+how\s+\w+\s+work", re.I), "research", 6.0),
    (re.compile(r"\btell\s+me\s+about\s+the\s+differences?\b", re.I), "research", 8.0),
    (re.compile(r"\bwhat'?s?\s+(are\s+)?(the\s+)?differences?\s+between\b", re.I), "research", 6.0),
    (re.compile(r"\bdifferences?\s+between\b", re.I), "research", 4.5),
    (re.compile(r"\b(search|look)\s+(for|up)\s+(papers?|research|best\s+practices?|studies|literature)\b", re.I), "research", 5.0),
    (re.compile(r"\bfind\s+(resources?|articles?|papers?|documentation)\s+(about|on|for)\b", re.I), "research", 5.0),
    (re.compile(r"\blook\s+up\s+(how|the\s+latest|the\s+documentation|what)\b", re.I), "research", 5.0),
    (re.compile(r"\btell\s+me\s+about\s+(the\s+)?(latest|recent|current)\s+(research|advances?|developments?)\b", re.I), "research", 4.0),
    (re.compile(r"\bwhat\s+(tools?|methods?|approaches?|techniques?|frameworks?)\s+(are|is)\s+(available|out there|commonly used)\b", re.I), "research", 4.0),
    (re.compile(r"\b(current\s+)?consensus\s+on\b", re.I), "research", 5.0),

    # Chat: opinion, emotion, casual patterns
    (re.compile(r"\b(what\s+do\s+you\s+think|tell\s+me\s+a\s+joke|how'?s\s+it\s+going)\b", re.I), "chat", 3.0),
    (re.compile(r"\b(what\s+should\s+i|help\s+me\s+(decide|come|settle|plan))\b", re.I), "chat", 2.5),
    (re.compile(r"\b(i'?m\s+(bored|feeling|thinking)|can\s+we\s+just\s+chat)\b", re.I), "chat", 3.0),
    (re.compile(r"\bgive\s+me\s+(your\s+)?(take|opinion|thoughts?)\s+(on|about)\b", re.I), "chat", 5.0),
    (re.compile(r"\bwhat'?s\s+your\s+take\s+on\b", re.I), "chat", 5.0),
    (re.compile(r"\bis\s+\w+\s+better\s+than\s+\w+\b", re.I), "chat", 5.5),
    (re.compile(r"\bshould\s+i\s+(learn|use|switch|choose|pick|go\s+with)\b", re.I), "chat", 4.0),
    (re.compile(r"\bwhat\s+are\s+some\s+(good|great|fun|interesting|creative)\b", re.I), "chat", 4.0),
    (re.compile(r"\bwrite\s+me\s+a\s+(cover\s+letter|bio|essay|poem|speech|toast|message|birthday)\b", re.I), "chat", 6.0),
    (re.compile(r"\bbook\s+recommendations?\b", re.I), "chat", 4.0),

    # Summarization: document comprehension signals
    (re.compile(r"^(summarize|condense|tl;?dr|recap|brief)\b", re.I), "summarization", 3.0),
    (re.compile(r"\b(key\s+points?|main\s+ideas?|gist|highlights)\b", re.I), "summarization", 2.0),
    (re.compile(r"\b(tl;?dr|too\s+long|shorten|condense)\b", re.I), "summarization", 2.0),
    (re.compile(r"\bwhat\b.{0,15}\b(this|these)\s+(whitepaper|document|report|paper|article|essay|transcript|chapter|email|thread|contract|policy)\b", re.I), "summarization", 4.0),
    (re.compile(r"\bwhat\s+(are|were)\s+the\s+(outcomes?|results?|conclusions?|action\s+items?|decisions?|themes?|arguments?)\b", re.I), "summarization", 3.5),
    (re.compile(r"\bwhat\s+does\s+this\s+(book|study|report|article)\s+say\b", re.I), "summarization", 6.0),
    (re.compile(r"\b(explain|help\s+me\s+(understand|make\s+sense\s+of))\s+(this|what\s+this)\b", re.I), "summarization", 6.0),
    (re.compile(r"\btell\s+me\s+what\s+this\b", re.I), "summarization", 5.0),
    (re.compile(r"\b(action\s+items?|lessons?\s+learned|core\s+arguments?)\s+(from|in)\b", re.I), "summarization", 6.0),
    (re.compile(r"\boutline\s+this\b", re.I), "summarization", 5.0),
    (re.compile(r"\bbreak\s+down\s+this\b", re.I), "summarization", 5.0),
    (re.compile(r"\bgive\s+me\s+(a\s+)?(brief|summary|overview|recap|digest)\s+(on|of)\s+(this|the)\b", re.I), "summarization", 6.0),
    (re.compile(r"\bkey\s+findings?\s+(from|of)\s+the\s+(latest|recent)\b", re.I), "summarization", 5.0),
    (re.compile(r"\b(extract|pull)\b.*\b(main|most\s+important|key)\s+(conclusion|takeaway|finding|point|insight)\b", re.I), "summarization", 5.0),

    # Code generation: structural output signals (code fences, definitions)
    (re.compile(r"```[\w]*\n"), "code_generation", 3.0),
    (re.compile(r"```"), "code_generation", 1.5),
    (re.compile(r"\bdef \w+\("), "code_generation", 3.0),
    (re.compile(r"\bfunction \w+\("), "code_generation", 3.0),
    (re.compile(r"\bclass \w+[:\({]"), "code_generation", 3.0),
    (re.compile(r"\bimport \w+"), "code_generation", 1.5),
    (re.compile(r"#include\s*<"), "code_generation", 3.0),

    # Code generation: syntax and language signals
    (re.compile(r"^(write|implement|create|build|generate|make)\b", re.I), "code_generation", 1.5),
    (re.compile(r"\b(python|typescript|javascript|rust|java|go|c\+\+|c#)\b", re.I), "code_generation", 2.0),
    (re.compile(r"\b(dockerfile|makefile|webpack|terraform|graphql)\b", re.I), "code_generation", 2.5),
    (re.compile(r"\b(async|await|decorator|middleware|endpoint|oauth|jwt|cors)\b", re.I), "code_generation", 1.5),
    (re.compile(r"\b(show\s+me\s+how\s+to|help\s+me\s+write\s+a?\s*(regex|function|script|code|class|program))\b", re.I), "code_generation", 3.5),
    (re.compile(r"\bbuild\s+a\s+\w+\s+pipeline\b", re.I), "code_generation", 5.0),
    (re.compile(r"\b(create|build|implement|write)\s+a\s+(tool|utility|library|scraper|script|program)\b", re.I), "code_generation", 4.0),
    (re.compile(r"\bi\s+need\s+(a|an)\s+\w+\s+(implementation|class|function|module|library|script)\b", re.I), "code_generation", 5.0),
    (re.compile(r"\bwhat\s+(pattern|approach|design|strategy)\s+should\s+i\s+use\b", re.I), "code_generation", 12.0),
    (re.compile(r"\bwhat'?s\s+the\s+best\s+way\s+to\s+(implement|build|code|write|create)\b", re.I), "code_generation", 7.0),
    (re.compile(r"\b(optimize|improve)\s+(this|my|the|a)\s+\w*\s*(query|code|function|algorithm)\b", re.I), "code_generation", 4.0),

    # Data extraction: structured data signals
    (re.compile(r"\b(extract|parse|scrape)\b.*\b(from|out of)\b.*\b(this|the|these)\s+(text|html|json|xml|csv|page|table|log|file|pdf|document)\b", re.I), "data_extraction", 5.0),
    (re.compile(r"\b(extract|get|grab)\s+all\s+\w+\s+from\b", re.I), "data_extraction", 4.0),
    (re.compile(r"\b(extract|get|pull)\s+(all\s+)?(the\s+)?(dates?|emails?|names?|numbers?|prices?|urls?|links?|addresses?|phone|ip)\b.*\b(from|in)\b", re.I), "data_extraction", 5.0),
    (re.compile(r"\b(extract|pull\s+out)\s+the\s+(numbers?|data|values?|figures?|statistics?|stats?)\b", re.I), "data_extraction", 7.0),
    (re.compile(r"\b(extract|get|pull)\s+all\s+(the\s+)?(function|method|class|variable|import)\s+(signatures?|names?|definitions?)\b", re.I), "data_extraction", 7.0),
    (re.compile(r"\bnormalize\s+the\b", re.I), "data_extraction", 3.0),
    (re.compile(r"\bflatten\s+the\b", re.I), "data_extraction", 3.0),

    # Agent planning: orchestration and multi-step reasoning signals
    (re.compile(r"\bplan\b", re.I), "agent_planning", 3.0),
    (re.compile(r"\bnext step\b", re.I), "agent_planning", 3.5),
    (re.compile(r"\bstrategy\b", re.I), "agent_planning", 3.0),
    (re.compile(r"\bworkflow\b", re.I), "agent_planning", 3.0),
    (re.compile(r"\borchestrat\w*\b", re.I), "agent_planning", 4.0),
    (re.compile(r"\bwhat should (i|we)\b", re.I), "agent_planning", 4.0),
    (re.compile(r"\bhow should (i|we)\b", re.I), "agent_planning", 4.0),
    (re.compile(r"\bbreak.?down\b", re.I), "agent_planning", 3.5),
    (re.compile(r"\bstep.?by.?step\b", re.I), "agent_planning", 3.5),

    # Scheduled job: automated recurring task signals
    (re.compile(r"\bdigest\b", re.I), "scheduled_job", 4.0),
    (re.compile(r"\bbriefing\b", re.I), "scheduled_job", 4.0),
    (re.compile(r"\bmorning\b", re.I), "scheduled_job", 3.0),
    (re.compile(r"\bdaily\b", re.I), "scheduled_job", 3.0),
    (re.compile(r"\bscheduled\b", re.I), "scheduled_job", 3.5),
    (re.compile(r"\bautomated\b", re.I), "scheduled_job", 3.5),
    (re.compile(r"\brecurring\b", re.I), "scheduled_job", 4.0),
    (re.compile(r"\bconsolidat\w*\b", re.I), "scheduled_job", 3.5),
    (re.compile(r"\blong.?term memory\b", re.I), "scheduled_job", 5.0),

    # Tool use: external service and action signals
    (re.compile(r"\b(send\s+(an?\s+)?email|check\s+(my\s+)?calendar|book\s+(a\s+)?meeting)\b", re.I), "tool_use", 4.0),
    (re.compile(r"\b(deploy\s+(the\s+|to\s+)|trigger\s+the|post\s+(this|to)\s+(the\s+)?slack)\b", re.I), "tool_use", 4.0),
    (re.compile(r"\bcreate\s+(a\s+)?(new\s+)?(ticket|issue|task|card|incident)\s+(in|on)\s+(jira|github|gitlab|linear|asana|trello|notion)\b", re.I), "tool_use", 5.0),
    (re.compile(r"\b(set\s+up|restart|start|stop)\s+(a\s+|the\s+)?(webhook|server|service|pod|container|instance|cluster)\b", re.I), "tool_use", 4.0),
    (re.compile(r"\b(create|schedule)\s+(a\s+)?(calendar\s+event|meeting|reminder|appointment)\b", re.I), "tool_use", 5.0),
    (re.compile(r"\brestart\s+(the\s+)?(production|staging|kubernetes|docker|server|pods?)\b", re.I), "tool_use", 5.0),
]

_RESEARCH_QUESTION_WORDS: frozenset = frozenset(["what", "how", "which", "compare", "vs", "versus"])
_RESEARCH_TECHNICAL_TERMS: frozenset = frozenset([
    "architectur", "framework", "techniqu", "approach", "methodolog",
    "ecosystem", "landscap", "tradeoff", "tradeoffs", "perform",
    "model", "algorithm", "benchmark", "infer", "train", "deploy",
    "strategi", "accuraci", "implication", "consensus", "prun", "find",
    "direction", "problem", "challeng", "pattern", "evalu", "analyz",
    "implement", "optim", "compress", "quantiz", "finetun", "lora",
    "qlora", "embed", "vector", "llm", "transformer", "neural",
    "tokeni", "prompt", "rag", "retrieval", "generat", "diffus",
])


# ---------------------------------------------------------------------------
# BM25+ Classifier
# ---------------------------------------------------------------------------


class BM25PlusClassifier:
    """BM25+ scoring with keyword expansion per category."""

    def __init__(
        self,
        categories: List[Dict[str, str]],
        keywords: Dict[str, List[str]],
        k1: float = 1.2,
        b: float = 0.0,
        delta: float = 0.5,
    ) -> None:
        self.cat_names = [c["name"] for c in categories]
        self.k1 = k1
        self.b = b
        self.delta = delta

        # Build category documents from name + description + keywords
        # High-signal keywords are repeated for higher TF
        self.cat_docs: Dict[str, List[str]] = {}
        for cat in categories:
            name = cat["name"]
            desc = cat.get("description", "")
            kws = keywords.get(name, [])
            high = HIGH_SIGNAL.get(name, [])
            doc_text = f"{name} {desc} {' '.join(kws)} {' '.join(high)}"
            self.cat_docs[name] = tokenize(doc_text)

        # Stem category docs
        self.cat_docs_stemmed: Dict[str, List[str]] = {
            name: [simple_stem(t) for t in tokens]
            for name, tokens in self.cat_docs.items()
        }

        # Compute average document length and IDF on stemmed docs
        all_docs = list(self.cat_docs_stemmed.values())
        self.avgdl = sum(len(d) for d in all_docs) / len(all_docs) if all_docs else 1.0
        self.n_docs = len(all_docs)
        self.df: Dict[str, int] = {}
        for doc in all_docs:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.entropy_weights = self._compute_entropy_weights()

    def _compute_entropy_weights(self) -> Dict[str, float]:
        """Entropy-based term weights: terms in fewer categories score higher."""
        term_cat_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for cat, tokens in self.cat_docs_stemmed.items():
            for t in tokens:
                term_cat_freq[t][cat] += 1

        weights: Dict[str, float] = {}
        max_entropy = math.log(self.n_docs) if self.n_docs > 1 else 1.0
        for term, cat_freqs in term_cat_freq.items():
            total = sum(cat_freqs.values())
            entropy = 0.0
            for count in cat_freqs.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log(p)
            weights[term] = 1.0 + (max_entropy - entropy) / max_entropy
        return weights

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        doc_len = len(doc_tokens)
        tf_map: Dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        score = 0.0
        for qt in set(query_tokens):
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            df = self.df.get(qt, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            entropy_w = self.entropy_weights.get(qt, 1.0)
            tf_norm = tf / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + tf)
            score += idf * entropy_w * (tf_norm + self.delta)
        return score

    def classify(self, text: str) -> str:
        tokens = tokenize(text)
        stemmed = [simple_stem(t) for t in tokens]

        # First-verb bonus
        first_bonuses: Dict[str, float] = defaultdict(float)
        if tokens:
            first = tokens[0]
            for cat, verbs in FIRST_VERB_BONUS.items():
                if first in verbs:
                    first_bonuses[cat] = 2.0

        # BM25+ scores
        scores: Dict[str, float] = {}
        for name in self.cat_names:
            doc = self.cat_docs_stemmed[name]
            score = self._bm25_score(stemmed, doc)
            penalty = sum(
                self.delta * 0.5 for t in tokens if t in NEGATIVE_KEYWORDS.get(name, [])
            )
            scores[name] = score - penalty + first_bonuses.get(name, 0.0)

        # Pattern bonuses
        _apply_pattern_bonuses(text, scores)

        # Research vs chat disambiguation
        stemmed_set = set(stemmed)
        tokens_set = set(tokens)
        if tokens_set & _RESEARCH_QUESTION_WORDS and stemmed_set & _RESEARCH_TECHNICAL_TERMS:
            scores["chat"] = scores.get("chat", 0.0) - 3.0

        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "uncategorized"


def _apply_pattern_bonuses(text: str, scores: Dict[str, float]) -> None:
    """Apply regex-based pattern bonuses to category scores."""
    text_lower = text.lower()
    for pattern, cat, bonus in PATTERN_BONUSES:
        if cat in scores and pattern.search(text_lower):
            scores[cat] += bonus


# ---------------------------------------------------------------------------
# Singleton classifier for default categories
# ---------------------------------------------------------------------------

_default_classifier: BM25PlusClassifier | None = None


def _get_default_classifier() -> BM25PlusClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = BM25PlusClassifier(DEFAULT_CATEGORIES, CATEGORY_KEYWORDS)
    return _default_classifier


def _get_classifier(categories: List[Dict[str, str]]) -> BM25PlusClassifier:
    """Return a classifier for the given categories. Uses cached instance for defaults."""
    normalized = [
        {**c, "name": c["name"].replace(" ", "_")} for c in categories
    ]
    default_names = {c["name"] for c in DEFAULT_CATEGORIES}
    cat_names = {c["name"] for c in normalized}

    if cat_names == default_names:
        return _get_default_classifier()

    return BM25PlusClassifier(normalized, CATEGORY_KEYWORDS)


def _classify_single(text: str, categories: List[Dict[str, str]]) -> str:
    """Classify a single text into the best matching category."""
    return _get_classifier(categories).classify(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_composite_text(entry: dict) -> Optional[str]:
    """Build a rich text representation of the entry for classification."""
    parts = []

    # User message (primary signal)
    preview = entry.get("user_message_preview")
    if preview:
        parts.append(preview)

    # Tool names — repeat 3x for TF weight boost, prefix with marker
    tool_names = entry.get("tool_names") or []
    if tool_names:
        joined = " ".join(tool_names)
        parts.append(f"tools {joined} {joined} {joined}")

    # has_tool_results flag — inject as text signal
    if entry.get("has_tool_results"):
        parts.append("tool result tool use action execute")

    # Assistant preview (what the model actually produced)
    asst = entry.get("assistant_preview")
    if asst:
        parts.append(asst)

    return " ".join(parts) if parts else None


def classify_entries(
    entries: List[Dict[str, Any]],
    categories: List[Dict[str, str]],
    entry_costs: List[float],
    entry_tokens: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Classify entries into categories using BM25+ scoring.

    Returns list of category dicts with cost aggregates, sorted by cost desc.
    Entries without user_message_preview are excluded from classification.
    If categories is empty, uses DEFAULT_CATEGORIES.
    """
    if not categories:
        categories = DEFAULT_CATEGORIES

    classifiable_indices = [
        (i, _build_composite_text(entries[i]))
        for i in range(len(entries))
        if _build_composite_text(entries[i])
    ]
    classifiable_set = {i for i, _ in classifiable_indices}

    clf = _get_classifier(categories)

    category_costs_map: Dict[int, float] = {i: 0.0 for i in range(len(categories))}
    category_calls_map: Dict[int, int] = {i: 0 for i in range(len(categories))}
    category_tokens_map: Dict[int, int] = {i: 0 for i in range(len(categories))}
    uncategorized_cost = 0.0
    uncategorized_calls = 0
    uncategorized_tokens = 0

    # Entries with no classifiable text go straight to uncategorized
    for i in range(len(entries)):
        if i not in classifiable_set:
            uncategorized_cost += entry_costs[i]
            uncategorized_calls += 1
            uncategorized_tokens += entry_tokens[i] if entry_tokens is not None else 0

    if not classifiable_indices:
        if uncategorized_calls == 0:
            return []
        total_cost = sum(entry_costs)
        return [
            {
                "name": "(uncategorized)",
                "cost_usd": round(uncategorized_cost, 6),
                "calls": uncategorized_calls,
                "tokens": uncategorized_tokens,
                "pct": round(uncategorized_cost / total_cost * 100, 1) if total_cost > 0 else 0.0,
            }
        ]

    cat_name_to_idx = {c["name"]: i for i, c in enumerate(categories)}

    for entry_idx, composite_text in classifiable_indices:
        cost = entry_costs[entry_idx]
        tokens = entry_tokens[entry_idx] if entry_tokens is not None else 0
        label = clf.classify(composite_text)

        if label in cat_name_to_idx:
            idx = cat_name_to_idx[label]
            category_costs_map[idx] += cost
            category_calls_map[idx] += 1
            category_tokens_map[idx] += tokens
        else:
            uncategorized_cost += cost
            uncategorized_calls += 1
            uncategorized_tokens += tokens

    total_cost = sum(entry_costs)

    result = []
    for i, cat in enumerate(categories):
        if category_calls_map[i] == 0:
            continue
        cost = category_costs_map[i]
        result.append(
            {
                "name": cat["name"],
                "cost_usd": round(cost, 6),
                "calls": category_calls_map[i],
                "tokens": category_tokens_map[i],
                "pct": round(cost / total_cost * 100, 1) if total_cost > 0 else 0.0,
            }
        )

    if uncategorized_calls > 0:
        result.append(
            {
                "name": "(uncategorized)",
                "cost_usd": round(uncategorized_cost, 6),
                "calls": uncategorized_calls,
                "tokens": uncategorized_tokens,
                "pct": round(uncategorized_cost / total_cost * 100, 1) if total_cost > 0 else 0.0,
            }
        )

    result.sort(key=lambda x: -x["cost_usd"])
    return result


async def classify_entries_async(
    entries: List[Dict[str, Any]],
    categories: List[Dict[str, str]],
    entry_costs: List[float],
    entry_tokens: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Async wrapper — runs classify_entries in a thread pool executor."""
    import functools
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(classify_entries, entries, categories, entry_costs, entry_tokens=entry_tokens)
    )
