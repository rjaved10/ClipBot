"""Run once: opens the bot's browser so you can log in to clipping.net. The bot reuses this login."""
import poster

if __name__ == "__main__":
    poster.login("https://clipping.net/", "clipping.net")
