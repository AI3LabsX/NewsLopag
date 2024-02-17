import os
import psycopg2
from tg.utils.logger import logger

DATABASE_URL = os.environ.get("DATABASE_URL")


def init_db():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS UserSettings (
                user_id BIGINT PRIMARY KEY,
                rss_feeds TEXT,
                topics TEXT
            )
            """
        )
        cursor.execute(
            """
            ALTER TABLE UserSettings
            ALTER COLUMN user_id TYPE BIGINT
            USING user_id::BIGINT
            """
        )
    conn.commit()
    logger.info("Database initialized and table altered if necessary.")
    conn.close()


async def get_topics(user_id):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT topics FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return row[0].split(',')
        else:
            return []
    finally:
        cursor.close()
        conn.close()


async def add_topic_to_db(user_id, topic):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT topics FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            new_topics = row[0] + ',' + topic
        else:
            new_topics = topic
        cursor.execute(
            "INSERT INTO UserSettings (user_id, topics) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET topics = %s",
            (user_id, new_topics, new_topics)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


async def delete_topic_from_db(user_id, topic):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT topics FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            topics = row[0].split(',')
            if topic in topics:
                topics.remove(topic)
                new_topics = ','.join(topics)
                cursor.execute(
                    "UPDATE UserSettings SET topics = %s WHERE user_id = %s",
                    (new_topics, user_id)
                )
                conn.commit()
    finally:
        cursor.close()
        conn.close()


async def get_rss_feeds(user_id):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT rss_feeds FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return row[0].split(',')
        else:
            return []
    finally:
        cursor.close()
        conn.close()


async def add_rss_feed_to_db(user_id, rss_feed):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT rss_feeds FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            new_rss_feeds = row[0] + ',' + rss_feed
        else:
            new_rss_feeds = rss_feed
        cursor.execute(
            "INSERT INTO UserSettings (user_id, rss_feeds) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET rss_feeds = %s",
            (user_id, new_rss_feeds, new_rss_feeds)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


async def delete_rss_feed_from_db(user_id, rss_feed):
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT rss_feeds FROM UserSettings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            rss_feeds = row[0].split(',')
            if rss_feed in rss_feeds:
                rss_feeds.remove(rss_feed)
                new_rss_feeds = ','.join(rss_feeds)
                cursor.execute(
                    "UPDATE UserSettings SET rss_feeds = %s WHERE user_id = %s",
                    (new_rss_feeds, user_id)
                )
                conn.commit()
    finally:
        cursor.close()
        conn.close()
