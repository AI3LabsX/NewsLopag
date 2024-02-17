"""
Module for reading values from environment variables in .env files
Contains Environment class, which performs reading of values and env - object of Environment class

Example:
    Import an instance of the class into the desired module:
        from tgbot.utils.environment import env

    Get the values from the environment variables:
        bot_token: str = env.get_token_or_exit()
        bot_admins: tuple[int, ...] = env.get_admin_ids_or_exit()
"""
import os
from sys import exit as sys_exit

from tg.utils.logger import logger


class Environment:
    @staticmethod
    def _get_env_var(var_name: str) -> str:
        value = os.environ.get(var_name)
        if value is None:
            logger.critical(f"{var_name} not found")
            sys_exit(1)
        return value

    def get_bot_token(self) -> str:
        return self._get_env_var("BOT_TOKEN")

    def get_admin_ids_or_exit(self) -> tuple[int, ...]:
        admin_ids_str = self._get_env_var(
            "ADMINS"
        )  # Assuming BOT_ADMINS is the environment variable name
        try:
            return tuple(map(int, admin_ids_str.split(",")))
        except ValueError:
            logger.critical(
                "Invalid format for BOT_ADMINS. It should be a comma-separated list of integers."
            )
            sys_exit(1)


env: Environment = Environment()
