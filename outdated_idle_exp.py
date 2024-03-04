from random import choices, uniform
from time import perf_counter, sleep

import click
from loguru import logger

from game_controller import GameController
from settings import (
    RANDOM_TURN_PROB,
    SPOT_DELAY,
    USE_BOOSTERS_AFTER,
    USE_PASSIVE_SKILLS_AFTER,
)
from utils import set_logger_level


@click.command()
@click.option("--debug", is_flag=True, help="Enable debug mode.")
def main(debug):
    if debug:
        set_logger_level("DEBUG")
    set_logger_level("INFO")
    run()


def run():
    game = GameController(start_delay=5)
    last_booster_act_time = perf_counter()
    last_passive_skills_act_time = perf_counter()

    game.toggle_passive_skills()
    game.mount()
    game.use_boosters()

    game.start_attack()
    while game.is_running:
        game.lure_many()

        if perf_counter() - last_booster_act_time >= USE_BOOSTERS_AFTER:
            last_booster_act_time = perf_counter()

            game.use_boosters()

        if perf_counter() - last_passive_skills_act_time >= USE_PASSIVE_SKILLS_AFTER:
            last_passive_skills_act_time = perf_counter()

            game.stop_attack()
            game.unmount()
            game.toggle_passive_skills()
            game.mount()
            game.start_attack()

        if choices([True, False], weights=[RANDOM_TURN_PROB, 1 - RANDOM_TURN_PROB]):
            game.turn_randomly()

        sleep(uniform(SPOT_DELAY - 1, SPOT_DELAY + 1))

        game.pickup_many()

    game.stop_attack()
    game.unmount()


if __name__ == '__main__':
    main()
    logger.success("Bot terminated.")
