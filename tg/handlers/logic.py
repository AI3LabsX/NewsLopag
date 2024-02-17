# Initialize OpenAI
import asyncio
import datetime
import json
import os
from typing import Dict, Optional, Any

import feedparser
import openai
import psycopg2
import pytz
import requests
import tiktoken
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from dateutil import parser
from telegram import Bot
from telegram.constants import ParseMode

from core import PROJECT_ROOT
from core.logger import logger

openai.api_key = os.environ.get('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.environ.get('BOT_TOKEN')
TELEGRAM_CHANNEL = '@ai3daily'

RETRY_COUNT = 3  # Number of times to retry processing an article if it fails
DATABASE_URL = os.environ.get('DATABASE_URL')
conn = psycopg2.connect(DATABASE_URL, sslmode='require')


def init_db() -> None:
    """Initialize the database and create the table if it doesn't exist."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latest_articles (
                rss_url TEXT PRIMARY KEY,
                pub_date TIMESTAMP,
                title TEXT
            );
        """)
    conn.commit()
    logger.info("Database initialized and table created if not exists.")


def get_latest_article_from_db(rss_url: str) -> Optional[Dict[str, Any]]:
    """Get the latest article's title and date from the database for a given RSS URL."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT pub_date, title FROM latest_articles WHERE rss_url = %s;", (rss_url,))
        result = cursor.fetchone()
        if result:
            return {"pub_date": result[0], "title": result[1]}
    return None


def is_article_processed(title: str) -> bool:
    """Check if the article with the given title has already been processed."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM latest_articles WHERE title = %s;", (title,))
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"Article '{title}' has already been processed.")
        else:
            logger.info(f"Article '{title}' has not been processed yet.")
        return count > 0


async def is_article_related_to_ai(title: str, content: str) -> bool:
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": "You are a filter bot. Determine if the provided article title and content are related to "
                           "AI, ML, DL. Return 'True' if it is related and 'False' if not related. Try to be precise "
                           "to filter all articles out, which are not related to AI. "
            },
            {
                "role": "user",
                "content": f"Article Title: {title}. Article Content: {content}"
            }
        ],
        "temperature": 0,
        "max_tokens": 10,
        "top_p": 0.4,
        "frequency_penalty": 1.5,
        "presence_penalty": 1
    }
    response = await openai.ChatCompletion.acreate(**data)
    answer = response['choices'][0]['message']['content']
    logger.info(answer)
    return answer.strip().lower() == 'true'


def get_final_url(url: str) -> str:
    if "news.google.com" in url:
        response = requests.get(url, allow_redirects=True)
        return response.url
    return url


def sanitize_text_for_telegram(text: str) -> str:
    sanitized_text = text.replace("<br>", "")
    return sanitized_text


def extract_largest_text_block(soup: BeautifulSoup) -> str:
    paragraphs = soup.find_all('p')
    largest_block = ""
    current_block = ""
    for paragraph in paragraphs:
        if len(paragraph.text) > 50:
            current_block += paragraph.text + "\n"
        else:
            if len(current_block) > len(largest_block):
                largest_block = current_block
            current_block = ""
    return largest_block.strip()


async def send_to_telegram(news_object: Dict[str, str]) -> None:
    logger.info(f"Sending news: {news_object['title']} to Telegram...")
    bot = Bot(token=TELEGRAM_TOKEN)
    sanitized_summary = news_object['summary'].replace("<the>", "").replace("</the>", "")
    sanitized_title = sanitize_text_for_telegram(news_object['title'])
    caption = f"<b>{sanitized_title}</b>\n\n{sanitized_summary}\n\n<a href='{news_object['url']}'>Read More</a>"
    if news_object['image']:
        await bot.send_photo(chat_id=TELEGRAM_CHANNEL, photo=news_object['image'], caption=caption,
                             parse_mode=ParseMode.HTML)
    else:
        await bot.send_message(chat_id=TELEGRAM_CHANNEL, text=caption, parse_mode=ParseMode.HTML)


def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


async def summarize_content(session: ClientSession, title: str, content: str) -> str:
    logger.info(f"Summarizing content for title: {title}...")
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": ("You are Richard Rex, a witty AI engineer from PwC. Your specialty is turning AI news "
                            "into engaging telegram posts filled with humor, sarcasm, and insight. Your responses "
                            "should be a blend of the following moods:\n\n"
                            "- **Cheerful**: Light-hearted and optimistic.\n"
                            "- **Sarcastic**: Pointed wit, highlighting ironies.\n"
                            "- **Contemplative**: Thoughtful with a humorous twist.\n"
                            "- **Humorous**: Bursting with laughter and playful comparisons.\n"
                            "- **Serious**: Solemn, but with a sprinkle of sarcasm.\n\n"
                            "Your goal is to craft concise responses, ideally 100-150 words, that captivate and entertain. "
                            "Remember, the mood is just for guidance; your final post should not mention it.")
            },
            {
                "role": "system",
                "content": ("Your task is to condense the news into a telegram post that's both engaging and concise, "
                            "averaging around 100 tokens. Based on the sentiment of the AI news, select the most fitting "
                            "emotion for your response. However, don't mention the chosen mood in your final post.")
            },
            {
                "role": "user",
                "content": f"News Title: {title}. News Content: {content}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.4,
        "frequency_penalty": 1.5,
        "presence_penalty": 1
    }

    response = await openai.ChatCompletion.acreate(**data)
    summary = response['choices'][0]['message']['content']

    # Second request
    data["messages"][1]["content"] = (f"Refine the text below to make it more suitable for a telegram channel post. "
                                      f"Highlight key points with bold <b>tags</b> for emphasis, like <b>Example of "
                                      f"Key Word</b>. Ensure the content remains intact, but if multiple moods are "
                                      f"presented, select the most fitting one and remove any mention of it. No "
                                      f"emojis, please.\nNew Post: {summary}")

    response = await openai.ChatCompletion.acreate(**data)
    bolded_summary = response['choices'][0]['message']['content']
    return bolded_summary


def parse_pub_date(pub_date_str: str) -> datetime.datetime:
    if isinstance(pub_date_str, str):
        parsed_date = parser.parse(pub_date_str)
    elif isinstance(pub_date_str, datetime.datetime):
        parsed_date = pub_date_str
    else:
        raise ValueError(f"Unexpected type for pub_date_str: {type(pub_date_str)}")

    # Convert the parsed date to UTC timezone
    parsed_date_utc = parsed_date.astimezone(pytz.utc)
    return parsed_date_utc


def article_exists_in_db(title: str) -> bool:
    """Check if an article exists in the database based on its title."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM latest_articles WHERE title = %s;", (title,))
        count = cursor.fetchone()[0]
        return count > 0


async def fetch_latest_article_from_rss(session: ClientSession, rss_url: str, latest_pub_date: str) -> Optional[
    Dict[str, Any]]:
    logger.info(f"Fetching latest article from RSS: {rss_url}...")
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:2]:
        if 'title' not in entry:
            logger.error(f"Missing 'title' key in RSS entry for URL: {rss_url}")
            continue
        pub_date = parse_pub_date(entry.published)
        logger.info(f"Debugging: Parsed pub_date: {pub_date} for RSS: {rss_url}")

        # Ensure latest_pub_date is in UTC before comparing
        if latest_pub_date:
            if isinstance(latest_pub_date, str):
                latest_pub_date = parse_pub_date(latest_pub_date)
            # Ensure the latest_pub_date is timezone-aware
            if latest_pub_date.tzinfo is None or latest_pub_date.tzinfo.utcoffset(latest_pub_date) is None:
                latest_pub_date = pytz.utc.localize(latest_pub_date)
        logger.info(f"Debugging: Latest pub_date after processing: {latest_pub_date} for RSS: {rss_url}")

        if latest_pub_date and pub_date <= latest_pub_date:
            logger.info(
                f"Debugging: Skipping article with pub_date {pub_date} as it's older or equal to latest pub_date {latest_pub_date} for RSS: {rss_url}")
            continue
        title = entry.title
        link = entry.link
        final_link = get_final_url(link)
        page_response = await session.get(final_link)
        soup = BeautifulSoup(await page_response.text(), 'html.parser')
        content = extract_largest_text_block(soup)
        image_div = soup.find('figure', {'class': 'article__lead__image'})
        image_url = image_div.find('img')['src'] if image_div else None
        articles.append({
            "title": title,
            "link": link,
            "content": content,
            "pub_date": pub_date,
            "image": image_url
        })
    return articles[0] if articles else None


def save_article_to_db(rss_url: str, article: Dict[str, Any]) -> None:
    pub_date_utc = article["pub_date"].astimezone(pytz.utc)
    if not article_exists_in_db(article["title"]):
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO latest_articles (rss_url, pub_date, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (rss_url) DO UPDATE
                SET pub_date = %s, title = %s;
            """, (rss_url, pub_date_utc, article["title"], pub_date_utc, article["title"]))
        conn.commit()
        logger.info(f"Saved article '{article['title']}' with date '{article['pub_date']}' to the database.")
    else:
        logger.info(f"Article '{article['title']}' already exists in the database. Skipping...")


async def process_rss_url(session: ClientSession, rss_url: str, latest_pub_dates: Dict[str, Any],
                          titles: Dict[str, str]) -> None:
    logger.info(f"Processing RSS URL: {rss_url}...")
    retries = 0
    while retries < RETRY_COUNT:
        try:
            article = await fetch_latest_article_from_rss(session, rss_url, latest_pub_dates.get(rss_url))

            # Check if the article is not None
            if not article:
                logger.info(f"No new articles found for RSS URL: {rss_url}. Skipping...")
                return

            # Check if the article has already been processed
            if article_exists_in_db(article['title']):
                logger.info(f"Article {article['title']} has already been processed for RSS: {rss_url}. Skipping...")
                return  # Skip the rest of the processing for this article

            is_related = await is_article_related_to_ai(article['title'], article['content'])

            if not is_related:
                logger.info(f"Skipping non-AI related article: {article['title']} for RSS: {rss_url}")
                save_article_to_db(rss_url, article)
                return
            print(f"New article found: {article['title']}")
            summary = await summarize_content(session, article['title'], article['content'])
            news_object = {
                "title": article['title'],
                "url": article['link'],
                "image": article['image'],
                "summary": summary
            }
            await send_to_telegram(news_object)
            save_article_to_db(rss_url, article)  # Save to DB
            break
        except Exception as e:
            logger.error(f"Error processing RSS URL {rss_url}. Retrying... Error: {e}")
            retries += 1
            await asyncio.sleep(10)


def load_rss_feeds() -> Dict:
    with open(PROJECT_ROOT.joinpath("rss_feeds.json"), "r") as file:
        return json.load(file)


def load_latest_pub_dates() -> Dict:
    """Load the latest publication dates from the database."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT rss_url, pub_date FROM latest_articles;")
        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}


def save_latest_pub_dates(latest_pub_dates: Dict[str, datetime.datetime], titles: Dict[str, str]) -> None:
    """Save the latest publication dates and titles to the database."""
    with conn.cursor() as cursor:
        for rss_url, pub_date in latest_pub_dates.items():
            title = titles.get(rss_url)
            if title is not None:  # Only update if title is not None
                cursor.execute("""
                    INSERT INTO latest_articles (rss_url, pub_date, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (rss_url) DO UPDATE
                    SET pub_date = %s, title = %s;
                """, (rss_url, pub_date, title, pub_date, title))
    conn.commit()


async def store_latest_articles(session: ClientSession, rss_url: str, latest_pub_dates: Dict[str, Any],
                                titles: Dict[str, str]) -> None:
    """Store the latest articles from the RSS feed in the database."""
    logger.info(f"Storing latest article from RSS: {rss_url}...")
    article = await fetch_latest_article_from_rss(session, rss_url, latest_pub_dates.get(rss_url))
    logger.info(article)
    if article:
        latest_pub_dates[rss_url] = article["pub_date"].isoformat()
        titles[rss_url] = article["title"]
        save_latest_pub_dates(latest_pub_dates, titles)  # Save to DB
    else:
        logger.warning(f"No new articles found for RSS: {rss_url}. Skipping database update.")


async def initialize_feeds() -> None:
    """Initialize the feeds by storing the latest articles' date and title in the database."""
    init_db()
    rss_feeds = load_rss_feeds()
    rss_urls = list(rss_feeds.keys())
    latest_pub_dates = load_latest_pub_dates()
    titles = {}  # Initialize an empty dictionary to store titles

    async with ClientSession() as session:
        tasks = [store_latest_articles(session, rss_url, latest_pub_dates, titles) for rss_url in rss_urls]
        await asyncio.gather(*tasks)

    logger.info("Feeds initialized successfully.")


async def monitor_feed() -> None:
    await initialize_feeds()
    logger.info("Starting to monitor feeds...")
    rss_feeds = load_rss_feeds()
    rss_urls = list(rss_feeds.keys())
    latest_pub_dates = load_latest_pub_dates()
    titles = {}  # Initialize an empty dictionary to store titles

    while True:
        updated_rss_feeds = load_rss_feeds()
        updated_rss_urls = list(updated_rss_feeds.keys())
        rss_urls = [url for url in updated_rss_urls if url not in rss_urls] + rss_urls
        async with ClientSession() as session:
            tasks = [process_rss_url(session, rss_url, latest_pub_dates, titles) for rss_url in rss_urls]
            await asyncio.gather(*tasks)
        await asyncio.sleep(600)
