"""The module presents a function that changes the default list of commands for the bot"""

from telegram import BotCommand
from telegram.ext import Application


async def set_default_commands(application: Application) -> None:
    """
    This function is used to change the list of commands of the bot, for example you can set
    different commands depending on the language of the user or show certain commands only to administrators.
    Learn more: https://core.telegram.org/bots/api#setmycommands
    """
    commands: list[BotCommand] = [
        BotCommand(command="start", description="ℹ️ Bot Start"),
        BotCommand(command="settings", description="ℹ️ Bot Settings"),
        BotCommand(command="news_on", description="ℹ️ Switch On news"),
        BotCommand(command="news_off", description="ℹ️ Switch Off news"),

    ]
    await application.bot.set_my_commands(commands=commands, scope=None, language_code=None)
