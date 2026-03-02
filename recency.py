
import asyncio
import aiohttp
import json
import os
import re
import sqlite3
import argparse
import hashlib
import time
import feedparser
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")
SERPAPI_KEY        = os.getenv("SERPAPI_KEY", "")

DEFAULT_MODEL   = "anthropic/claude-3.5-sonnet"
DB_PATH         = "ai_trends.db"
OUTPUT_DIR      = Path("output")
RATE_LIMIT_DELAY = 1.5   # seconds between API calls
MAX_RETRIES      = 3

console = Console() if RICH_AVAILABLE else None

def log(msg: str, style: str = ""):
    if RICH_AVAILABLE:
        console.print(msg, style=style)
    else:
        print(msg)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS raw_articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            source TEXT,
            content TEXT,
            published TEXT,
            fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trends (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            category TEXT,
            score REAL,
            sources TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS linkedin_posts (
            id TEXT PRIMARY KEY,
            trend_title TEXT,
            post TEXT,
            model TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn

def cache_article(conn, title, url, source, content, published=""):
    article_id = hashlib.md5(url.encode()).hexdigest()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO raw_articles VALUES (?,?,?,?,?,?,?)",
            (article_id, title, url, source, content[:5000], published, datetime.now().isoformat())
        )
        conn.commit()
    except Exception:
        pass
    return article_id

def get_cached_articles(conn, days: int = 7):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT title, url, source, content, published FROM raw_articles WHERE fetched_at > ?",
        (cutoff,)
    ).fetchall()
    return [{"title": r[0], "url": r[1], "source": r[2], "content": r[3], "published": r[4]} for r in rows]

# ─────────────────────────────────────────────
# WEB SCRAPING & RSS
# ─────────────────────────────────────────────

RSS_FEEDS = {
    "arXiv AI":         "https://rss.arxiv.org/rss/cs.AI",
    "arXiv ML":         "https://rss.arxiv.org/rss/cs.LG",
    "arXiv CV":         "https://rss.arxiv.org/rss/cs.CV",
    "arXiv CL":         "https://rss.arxiv.org/rss/cs.CL",   # Computation & Language (NLP/LLMs)
    "arXiv RO":         "https://rss.arxiv.org/rss/cs.RO",   # Robotics + AI
    "MIT Tech Review":  "https://www.technologyreview.com/feed/",
    "VentureBeat AI":   "https://venturebeat.com/category/ai/feed/",
    "TechCrunch AI":    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "Google AI Blog":   "https://blog.research.google/feeds/posts/default?alt=rss",
    "Hugging Face":     "https://huggingface.co/blog/feed.xml",
    "OpenAI Blog":      "https://openai.com/blog/rss.xml",
    "The Batch":        "https://www.deeplearning.ai/the-batch/feed/",
    "Towards Data Science": "https://towardsdatascience.com/feed",
}

SCRAPE_URLS = {
    "GitHub Trending ML": "https://github.com/trending?since=weekly&spoken_language_code=&q=machine+learning",
    "Papers With Code":   "https://paperswithcode.com/latest",
    "Reddit r/MachineLearning": "https://www.reddit.com/r/MachineLearning/hot.json?limit=20",
    "Reddit r/artificial": "https://www.reddit.com/r/artificial/hot.json?limit=20",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AITrendBot/1.0; research purposes)",
    "Accept": "application/json, text/html",
    "Accept-Encoding": "gzip, deflate",   # explicitly exclude 'br' (brotli) — aiohttp can't decode it
}

async def fetch_rss(session: aiohttp.ClientSession, name: str, url: str, days: int) -> list[dict]:
    articles = []
    try:
        rss_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AITrendBot/1.0; research purposes)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Encoding": "gzip, deflate",  # never request brotli
        }
        async with session.get(url, headers=rss_headers, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True) as resp:
            raw = await resp.read()   # raw bytes — feedparser handles encoding detection
        feed = feedparser.parse(raw)
        cutoff = datetime.now() - timedelta(days=days)
        # Sort entries by date descending before filtering
        def parse_pub(e):
            p = e.get("published_parsed") or e.get("updated_parsed")
            return datetime(*p[:6]) if p else datetime.min
        sorted_entries = sorted(feed.entries, key=parse_pub, reverse=True)
        for entry in sorted_entries[:50]:
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            pub_dt = datetime(*pub[:6]) if pub else None
            if pub_dt is None or pub_dt < cutoff:
                continue
            content = entry.get("summary", "") or entry.get("description", "")
            soup = BeautifulSoup(content, "html.parser")
            clean = soup.get_text(separator=" ").strip()[:1000]
            hours_old = (datetime.now() - pub_dt).total_seconds() / 3600
            articles.append({
                "title":      entry.get("title", ""),
                "url":        entry.get("link", ""),
                "source":     name,
                "content":    clean,
                "published":  pub_dt.isoformat(),
                "hours_old":  round(hours_old, 1),
            })
    except Exception as e:
        log(f"  [yellow]RSS failed ({name}): {e}[/yellow]")
    return articles

async def fetch_reddit(session: aiohttp.ClientSession, name: str, url: str) -> list[dict]:
    articles = []
    try:
        reddit_headers = {
            "User-Agent": "AITrendBot/1.0 (research bot)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        async with session.get(url, headers=reddit_headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)
        posts = data.get("data", {}).get("children", [])
        for p in posts:
            d = p.get("data", {})
            if d.get("score", 0) < 100:
                continue
            articles.append({
                "title":     d.get("title", ""),
                "url":       f"https://reddit.com{d.get('permalink', '')}",
                "source":    name,
                "content":   d.get("selftext", "")[:800] or d.get("title", ""),
                "published": datetime.fromtimestamp(d.get("created_utc", time.time())).isoformat(),
            })
    except Exception as e:
        log(f"  [yellow]Reddit failed ({name}): {e}[/yellow]")
    return articles

async def fetch_papers_with_code(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch trending papers from HuggingFace Daily Papers (more reliable than PwC API)."""
    articles = []
    # Try HuggingFace Daily Papers API
    try:
        url = "https://huggingface.co/api/daily_papers?page=0&limit=20"
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)
        for p in (data if isinstance(data, list) else data.get("papers", [])):
            paper = p.get("paper", p)
            articles.append({
                "title":     paper.get("title", ""),
                "url":       f"https://huggingface.co/papers/{paper.get('id', '')}",
                "source":    "HuggingFace Daily Papers",
                "content":   paper.get("summary", "")[:800],
                "published": paper.get("publishedAt", datetime.now().isoformat()),
            })
        if articles:
            return articles
    except Exception as e:
        log(f"  [yellow]HF Daily Papers failed: {e}[/yellow]")

    # Fallback: PapersWithCode RSS feed
    try:
        url = "https://paperswithcode.com/api/v1/papers/?items_per_page=15&ordering=-published"
        pwc_headers = {**HEADERS, "Accept": "application/json"}
        async with session.get(url, headers=pwc_headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            text = await resp.text()
            data = json.loads(text)
        for p in data.get("results", []):
            articles.append({
                "title":     p.get("title", ""),
                "url":       f"https://paperswithcode.com{p.get('url_abs', '')}",
                "source":    "Papers With Code",
                "content":   p.get("abstract", "")[:800],
                "published": p.get("published", ""),
            })
    except Exception as e:
        log(f"  [yellow]PapersWithCode API failed: {e}[/yellow]")
    return articles

async def fetch_github_trending(session: aiohttp.ClientSession) -> list[dict]:
    articles = []
    topics = ["llm", "ai-agents", "machine-learning", "large-language-model", "generative-ai"]
    for topic in topics:
        try:
            url = f"https://github.com/trending?since=weekly&q={topic}"
            gh_headers = {**HEADERS, "Accept-Encoding": "gzip, deflate"}
            async with session.get(url, headers=gh_headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                html = await resp.text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for repo in soup.select("article.Box-row")[:5]:
                name_tag = repo.select_one("h2 a")
                desc_tag = repo.select_one("p")
                if name_tag:
                    repo_name = name_tag.get_text(strip=True).replace("\n", "").replace(" ", "")
                    desc = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({
                        "title":     f"Trending GitHub: {repo_name}",
                        "url":       f"https://github.com{name_tag.get('href', '')}",
                        "source":    "GitHub Trending",
                        "content":   desc,
                        "published": datetime.now().isoformat(),
                    })
        except Exception as e:
            log(f"  [yellow]GitHub trending failed ({topic}): {e}[/yellow]")
    return articles

async def fetch_huggingface_models(session: aiohttp.ClientSession) -> list[dict]:
    articles = []
    try:
        url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=20&full=true"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
        for m in data[:10]:
            articles.append({
                "title":     f"HF Model: {m.get('modelId', '')}",
                "url":       f"https://huggingface.co/{m.get('modelId', '')}",
                "source":    "Hugging Face Models",
                "content":   f"Downloads: {m.get('downloads', 0)} | Tags: {', '.join(m.get('tags', [])[:5])}",
                "published": m.get("lastModified", datetime.now().isoformat()),
            })
    except Exception as e:
        log(f"  [yellow]HuggingFace models failed: {e}[/yellow]")
    return articles

async def search_tavily(query: str, days: int) -> list[dict]:
    """Use Tavily for deep search if API key is available."""
    if not TAVILY_API_KEY:
        return []
    articles = []
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": 10,
            "days": days,
        }
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        data = resp.json()
        for r in data.get("results", []):
            articles.append({
                "title":     r.get("title", ""),
                "url":       r.get("url", ""),
                "source":    f"Tavily: {query}",
                "content":   r.get("content", "")[:800],
                "published": datetime.now().isoformat(),
            })
    except Exception as e:
        log(f"  [yellow]Tavily search failed: {e}[/yellow]")
    return articles

async def collect_all_articles(days: int, topics: list[str]) -> list[dict]:
    """Concurrently fetch from all sources."""
    all_articles = []
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        # RSS feeds
        for name, url in RSS_FEEDS.items():
            tasks.append(fetch_rss(session, name, url, days))

        # Reddit
        tasks.append(fetch_reddit(session, "Reddit ML", SCRAPE_URLS["Reddit r/MachineLearning"]))
        tasks.append(fetch_reddit(session, "Reddit AI", SCRAPE_URLS["Reddit r/artificial"]))

        # Special sources
        tasks.append(fetch_papers_with_code(session))
        tasks.append(fetch_github_trending(session))
        tasks.append(fetch_huggingface_models(session))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_articles.extend(r)

        # Tavily deep search for each topic
        if TAVILY_API_KEY:
            today = datetime.now().strftime("%B %Y")
            tavily_queries = [
                f"AI model release announcement {today}",
                f"new LLM launched {today}",
                f"AI research paper breakthrough {today}",
                f"AI startup funding raised {today}",
                f"open source AI model released {today}",
                f"AI regulation policy {today}",
                f"AI agent framework released {today}",
                "AI news today breaking",
            ] + [f"{t} AI latest release {today}" for t in topics]
            tavily_tasks = [search_tavily(q, days) for q in tavily_queries]
            tavily_results = await asyncio.gather(*tavily_tasks)
            for r in tavily_results:
                all_articles.extend(r)

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for a in all_articles:
        if a["url"] not in seen_urls and a["title"]:
            seen_urls.add(a["url"])
            unique.append(a)

    return unique

# ─────────────────────────────────────────────
# OPENROUTER API
# ─────────────────────────────────────────────

async def call_openrouter(prompt: str, system: str, model: str, retries: int = MAX_RETRIES) -> str:
    """Call OpenRouter API with retry logic."""
    if not OPENROUTER_API_KEY:
        return "[ERROR] No OPENROUTER_API_KEY set in .env"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-trend-researcher",
        "X-Title": "AI Trend Researcher",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 1200,
        "temperature": 0.85,
    }

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt * 5
                        log(f"  [yellow]Rate limited. Waiting {wait}s...[/yellow]")
                        await asyncio.sleep(wait)
                        continue
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"[ERROR] API call failed: {e}"
            await asyncio.sleep(2 ** attempt)
    return "[ERROR] Max retries exceeded"

# ─────────────────────────────────────────────
# TREND ANALYSIS
# ─────────────────────────────────────────────

CATEGORIES = {
    "LLM & Foundation Models": ["llm", "gpt", "claude", "gemini", "llama", "mistral", "language model", "foundation model", "transformer"],
    "AI Agents & Automation":  ["agent", "autonomous", "agentic", "workflow", "automation", "multi-agent", "tool use", "function calling"],
    "Multimodal AI":           ["multimodal", "vision", "image generation", "video", "audio", "speech", "stable diffusion", "dall-e", "sora"],
    "AI Infrastructure":       ["inference", "gpu", "chip", "hardware", "tpu", "cuda", "vllm", "triton", "serving", "quantization", "gguf"],
    "Open Source AI":          ["open source", "open-source", "hugging face", "weights", "apache", "mit license", "community model"],
    "AI Research":             ["paper", "arxiv", "benchmark", "eval", "dataset", "training", "fine-tuning", "rlhf", "alignment"],
    "AI Business & Funding":   ["funding", "raises", "valuation", "series", "startup", "acquisition", "partnership", "enterprise"],
    "AI Policy & Safety":      ["regulation", "policy", "safety", "alignment", "bias", "ethics", "eu ai act", "governance", "ban"],
}

def categorize_article(article: dict) -> str:
    text = (article["title"] + " " + article["content"]).lower()
    scores = {}
    for cat, keywords in CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General AI"

def score_article(article: dict) -> float:
    """Score article — recency is king."""
    score = 0.0
    try:
        pub = datetime.fromisoformat(article["published"][:19])
        hours_old = (datetime.now() - pub).total_seconds() / 3600
        # Exponential recency decay: <6h=20pts, <24h=15pts, <48h=10pts, <72h=5pts, older=0
        if hours_old < 6:    score += 20
        elif hours_old < 24: score += 15
        elif hours_old < 48: score += 10
        elif hours_old < 72: score += 5
        # Extra freshness bonus for truly breaking news
        if hours_old < 2:    score += 10
    except Exception:
        pass
    # Content richness (capped, secondary to recency)
    score += min(len(article.get("content", "")) / 300, 4)
    # Source quality bonus
    premium_sources = ["arxiv", "openai", "google ai", "hugging face", "daily papers", "mit tech", "deepmind", "anthropic"]
    if any(p in article["source"].lower() for p in premium_sources):
        score += 5
    # Deprioritize generic aggregators
    noise_sources = ["towards data science", "medium"]
    if any(p in article["source"].lower() for p in noise_sources):
        score -= 3
    return round(max(score, 0), 2)

async def synthesize_trends(articles: list[dict], model: str) -> list[dict]:
    """Use LLM to synthesize raw articles into structured weekly trends."""
    log("\n[bold cyan]🧠 Synthesizing trends with LLM...[/bold cyan]")

    # Group by category
    by_category: dict[str, list] = {}
    for a in articles:
        cat = categorize_article(a)
        by_category.setdefault(cat, []).append(a)

    trends = []
    for category, cat_articles in by_category.items():
        # Take top 15 by score
        # Sort by score (recency-weighted), take top 20 for LLM context
        top = sorted(cat_articles, key=score_article, reverse=True)[:20]
        # Only proceed if we have articles from the last 7 days
        if not top:
            continue
        if not top:
            continue

        articles_text = "\n".join([
            f"- [{a['source']}] {a['title']}: {a['content'][:200]}"
            for a in top
        ])

        system = "You are a senior AI research analyst. Respond ONLY in valid JSON."
        prompt = f"""You are analyzing BREAKING AI news from the past 7 days. Focus ONLY on the newest, most significant developments.

Category: "{category}"

RECENT ARTICLES (newest first):
{articles_text}

Identify the 1-2 most significant RECENT trends. Prioritize:
- Model/tool releases announced THIS WEEK (name them specifically)
- Research papers just published (cite the actual title)
- Major funding rounds just closed
- New APIs, frameworks, or open-source drops
- Regulatory moves just announced

Ignore evergreen background knowledge. If nothing is truly new this week, say so with impact: "Low".

Respond with a JSON array (max 2 objects):
{{
  "title": "Specific newsy title — name the actual product/model/company (max 12 words)",
  "summary": "2-3 sentences. Lead with WHAT happened and WHEN. Be specific with names, numbers, dates.",
  "key_points": ["specific fact with number or name", "another concrete fact", "why it matters now"],
  "impact": "High|Medium|Low",
  "category": "{category}",
  "sources": ["source1", "source2"],
  "recency": "breaking|this-week|recent"
}}

Return ONLY the JSON array, no other text."""

        result = await call_openrouter(prompt, system, model)
        await asyncio.sleep(RATE_LIMIT_DELAY)

        try:
            # Clean JSON
            clean = re.sub(r"```json|```", "", result).strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                for t in parsed:
                    t["score"] = sum(score_article(a) for a in top[:5]) / 5
                trends.extend(parsed)
        except Exception as e:
            log(f"  [yellow]Failed to parse trend for {category}: {e}[/yellow]")

    # Sort by impact then score
    impact_order = {"High": 3, "Medium": 2, "Low": 1}
    trends.sort(key=lambda t: (impact_order.get(t.get("impact", "Low"), 0), t.get("score", 0)), reverse=True)
    return trends[:10]  # Top 10 trends

# ─────────────────────────────────────────────
# LINKEDIN POST GENERATOR
# ─────────────────────────────────────────────

POST_STYLES = [
    ("analytical",    "data-driven, cite specific numbers and technical details, appeal to AI practitioners"),
    ("opinion",       "bold take or contrarian insight, provocative but grounded, appeal to AI thought leaders"),
    ("news_brief",    "breaking news style, concise punchy sentences, broad professional audience"),
    ("storytelling",  "start with a relatable scenario or analogy, human angle, appeal to business leaders"),
    ("how_it_works",  "educational explainer, demystify the tech, appeal to curious professionals new to AI"),
]

async def generate_linkedin_post(trend: dict, style_name: str, style_desc: str, model: str) -> str:
    key_points = "\n".join(f"- {p}" for p in trend.get("key_points", []))
    recency_label = trend.get("recency", "this-week")
    recency_note = {
        "breaking": "🚨 BREAKING — happened in last 24-48 hours",
        "this-week": "📅 THIS WEEK — very fresh",
        "recent":    "🕐 Last 7 days",
    }.get(recency_label, "this week")

    system = """You are a top LinkedIn content creator specializing in AI. 
You write viral posts that get 10K+ impressions. You understand the LinkedIn algorithm deeply.
Never use em-dashes. Write authentically, not like a press release."""

    prompt = f"""Write a high-engagement LinkedIn post about this AI trend using a "{style_name}" style ({style_desc}).

TREND: {trend['title']}
CATEGORY: {trend.get('category', 'AI')}
RECENCY: {recency_label} {recency_note}
SUMMARY: {trend['summary']}
KEY POINTS:
{key_points}
IMPACT: {trend.get('impact', 'High')}

IMPORTANT: This is fresh news. Write in present tense. Reference that this just happened.
Do NOT say "recently" or "in recent years" — be specific: "this week", "just released", "announced yesterday".

LINKEDIN POST REQUIREMENTS:
1. LINE 1 (Hook): Must be scroll-stopping. Bold claim, surprising fact, or powerful question. Max 12 words. Make people click "see more".
2. LINE 2: Empty line (for visual break)
3. BODY (3-5 short paragraphs): Tell the story. Use short sentences. 1-2 sentences per paragraph max. Mix insight with context.
4. KEY TAKEAWAYS: 3-4 bullet points with → emoji
5. ENGAGEMENT QUESTION: End with ONE thought-provoking question to drive comments
6. HASHTAGS: 5-7 relevant hashtags (mix of broad #AI and specific ones)

STYLE NOTES for {style_name}:
- {style_desc}
- No corporate jargon
- Sound like a smart human, not a bot
- No em-dashes

Write ONLY the post text, nothing else."""

    return await call_openrouter(prompt, system, model)

async def generate_all_posts(trends: list[dict], model: str) -> list[dict]:
    """Generate LinkedIn posts for each trend with varied styles."""
    posts = []
    log("\n[bold green]✍️  Generating LinkedIn posts...[/bold green]")

    for i, trend in enumerate(trends):
        style_name, style_desc = POST_STYLES[i % len(POST_STYLES)]
        log(f"  Generating post {i+1}/{len(trends)}: {trend['title'][:50]}...")
        post_text = await generate_linkedin_post(trend, style_name, style_desc, model)
        await asyncio.sleep(RATE_LIMIT_DELAY)

        posts.append({
            "trend_title": trend["title"],
            "category":    trend.get("category", ""),
            "impact":      trend.get("impact", ""),
            "style":       style_name,
            "post":        post_text,
            "model":       model,
            "created_at":  datetime.now().isoformat(),
        })

    return posts

# ─────────────────────────────────────────────
# OUTPUT FORMATTERS
# ─────────────────────────────────────────────

def save_trend_report(trends: list[dict], articles: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"trend_report_{date_str}.md"

    lines = [
        f"# 🤖 Weekly AI Trend Report — {date_str}",
        f"\n> Generated by AI Trend Researcher | {len(articles)} sources analyzed | {len(trends)} trends identified\n",
        "---\n",
        "## 📊 Trend Summary\n",
    ]

    # Summary table in Markdown
    lines.append("| # | Trend | Category | Impact |")
    lines.append("|---|-------|----------|--------|")
    for i, t in enumerate(trends, 1):
        lines.append(f"| {i} | {t['title']} | {t.get('category','')} | {t.get('impact','')} |")

    lines.append("\n---\n")
    lines.append("## 🔍 Deep Trend Analysis\n")

    for i, t in enumerate(trends, 1):
        lines.append(f"### {i}. {t['title']}")
        lines.append(f"**Category:** {t.get('category', '')} | **Impact:** {t.get('impact', '')} | **Score:** {t.get('score', 0):.1f}\n")
        lines.append(f"{t['summary']}\n")
        lines.append("**Key Points:**")
        for kp in t.get("key_points", []):
            lines.append(f"- {kp}")
        sources = t.get("sources", [])
        if sources:
            lines.append(f"\n**Sources:** {', '.join(sources)}")
        lines.append("\n---\n")

    # Source breakdown
    source_counts: dict = {}
    for a in articles:
        source_counts[a["source"]] = source_counts.get(a["source"], 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    lines.append("## 📰 Top Sources This Week\n")
    for src, count in top_sources:
        lines.append(f"- **{src}**: {count} articles")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def save_linkedin_posts(posts: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"linkedin_posts_{date_str}.md"

    lines = [
        f"# 💼 LinkedIn Posts — {date_str}",
        f"\n> {len(posts)} posts generated | Ready to copy-paste\n",
        "---\n",
    ]

    for i, p in enumerate(posts, 1):
        lines.append(f"## Post {i}: {p['trend_title']}")
        lines.append(f"**Style:** {p['style']} | **Category:** {p['category']} | **Impact:** {p['impact']}")
        lines.append(f"**Model:** {p['model']}\n")
        lines.append("```")
        lines.append(p["post"])
        lines.append("```")
        lines.append("\n---\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def save_json(trends: list[dict], posts: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"ai_trends_{date_str}.json"
    data = {
        "generated_at": datetime.now().isoformat(),
        "trends": trends,
        "linkedin_posts": posts,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def print_summary(trends: list[dict], posts: list[dict]):
    if not RICH_AVAILABLE:
        print(f"\n✅ Done! {len(trends)} trends found, {len(posts)} posts generated.")
        return

    console.print("\n")
    console.print(Panel.fit("[bold green]✅ AI Trend Research Complete![/bold green]"))

    table = Table(title="Top AI Trends This Week", show_lines=True)
    table.add_column("#",        style="cyan",  width=3)
    table.add_column("Trend",    style="white", width=35)
    table.add_column("Category", style="yellow", width=22)
    table.add_column("Impact",   style="green",  width=8)

    for i, t in enumerate(trends[:7], 1):
        impact = t.get("impact", "")
        color = "bold red" if impact == "High" else "yellow" if impact == "Medium" else "white"
        table.add_row(str(i), t["title"][:34], t.get("category","")[:21], f"[{color}]{impact}[/{color}]")

    console.print(table)
    console.print(f"\n[bold]Posts generated:[/bold] {len(posts)}")
    console.print(f"[bold]Output folder:[/bold]  ./{OUTPUT_DIR}/")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="AI Trend Researcher & LinkedIn Post Generator")
    parser.add_argument("--days",    type=int,   default=7,               help="Days to look back (default: 7)")
    parser.add_argument("--topics",  type=str,   default="LLM agents multimodal", help="Extra topics to search")
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL,   help="OpenRouter model to use")
    parser.add_argument("--no-posts", action="store_true",                help="Skip LinkedIn post generation")
    parser.add_argument("--use-cache", action="store_true",               help="Use cached articles from DB")
    args = parser.parse_args()

    topics = args.topics.split()

    log("\n[bold magenta]🚀 AI Trend Researcher & LinkedIn Post Generator[/bold magenta]")
    log(f"[dim]Model: {args.model} | Days: {args.days} | Topics: {topics}[/dim]\n")

    if not OPENROUTER_API_KEY:
        log("[bold red]⚠️  WARNING: OPENROUTER_API_KEY not set in .env — LLM steps will fail.[/bold red]")

    # Init DB
    conn = init_db()

    # Step 1: Collect articles
    if args.use_cache:
        log("[cyan]📦 Loading from cache...[/cyan]")
        articles = get_cached_articles(conn, args.days)
        log(f"  Loaded {len(articles)} cached articles")
    else:
        log("[cyan]🌐 Fetching articles from all sources...[/cyan]")
        articles = await collect_all_articles(args.days, topics)
        log(f"  ✅ Collected {len(articles)} unique articles")
        for a in articles:
            cache_article(conn, a["title"], a["url"], a["source"], a["content"], a["published"])

    if not articles:
        log("[red]No articles found. Check your internet connection.[/red]")
        return

    # Step 2: Synthesize trends
    trends = await synthesize_trends(articles, args.model)
    log(f"\n  ✅ Identified {len(trends)} trends")

    # Save trends to DB
    for t in trends:
        tid = hashlib.md5(t["title"].encode()).hexdigest()
        conn.execute(
            "INSERT OR REPLACE INTO trends VALUES (?,?,?,?,?,?,?)",
            (tid, t["title"], t.get("summary",""), t.get("category",""),
             t.get("score",0), json.dumps(t.get("sources",[])), datetime.now().isoformat())
        )
    conn.commit()

    # Step 3: Generate LinkedIn posts
    posts = []
    if not args.no_posts and trends:
        posts = await generate_all_posts(trends, args.model)
        for p in posts:
            pid = hashlib.md5((p["trend_title"] + p["created_at"]).encode()).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO linkedin_posts VALUES (?,?,?,?,?)",
                (pid, p["trend_title"], p["post"], p["model"], p["created_at"])
            )
        conn.commit()

    # Step 4: Save outputs
    report_path = save_trend_report(trends, articles)
    posts_path  = save_linkedin_posts(posts) if posts else None
    json_path   = save_json(trends, posts)

    log(f"\n[bold]📄 Trend Report:[/bold]  {report_path}")
    if posts_path:
        log(f"[bold]💼 LinkedIn Posts:[/bold] {posts_path}")
    log(f"[bold]🗂️  JSON Data:[/bold]     {json_path}")

    print_summary(trends, posts)
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
