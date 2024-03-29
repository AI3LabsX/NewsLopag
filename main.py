"""Launches the bot"""
from telegram.constants import ParseMode
from telegram.ext import AIORateLimiter, Application, Defaults

from tg.handlers import HANDLERS
from tg.handlers.db_tools import init_db

from tg.handlers.errors import error_handler
from tg.utils.bot_commands import set_default_commands
from tg.utils.environment import env
from tg.utils.logger import logger


async def on_startup(application: Application) -> None:
    """
    The function that runs when the bot starts, before the application.run_polling()
    In this example, the function sets the default commands for the bot
    """
    await set_default_commands(application=application)


def register_all_handlers(application: Application) -> None:
    """Registers handlers"""
    application.add_handlers(handlers=HANDLERS)  # type: ignore
    application.add_error_handler(callback=error_handler)


def start_bot() -> None:
    """Launches the bot"""

    # Create the Application and pass it your bot's token.
    application: Application = (
        Application.builder()
        .token(token=env.get_bot_token())
        .defaults(defaults=Defaults(parse_mode=ParseMode.HTML, block=False))
        .rate_limiter(rate_limiter=AIORateLimiter(max_retries=3))
        .post_init(post_init=on_startup)
        .build()
    )

    register_all_handlers(application=application)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    try:
        init_db()
        logger.info("Starting bot")
        start_bot()
    except Exception as exc:
        logger.critical("Unhandled error: %s", repr(exc))
    finally:
        logger.info("Bot stopped!")
