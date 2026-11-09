# Clip bot

Watches X for video posts about DDG, N3on, PlaqueBoyMax, JasonTheWeen and Lacy, then for each one:

1. Downloads it. Skips it if it's under 720p, shorter than 6 s or longer than 3 min.
2. **Skips any clip with someone else's watermark** (an @handle or TikTok/CapCut/clips text that stays on screen) or a headline already burned in.
3. Rewords the X post into a headline using the local Ollama model (free). Posts not about these creators are skipped.
4. Lays it out at 1080x1920. A reaction (wide video with a small facecam) gets the cam on top and the content on the bottom. Anything else sits in the middle over a blurred copy. The headline is in white boxes in TikTok Sans.
5. Posts it to TikTok. The description is the headline plus emoji plus the creator tags (#lacy #neon ...) plus #fyp #viral #funny #clips.

## One-time setup

1. **X login.** Install the "Get cookies.txt LOCALLY" extension in Chrome, open x.com while logged in, export, and save the file as `cookies/x_cookies.txt`.
   Use a spare X account if you can, because X can lock accounts that scrape.
2. **TikTok login.** Run `python login.py`, log in in the window that opens, then close it.
3. Make sure Ollama is running (it starts with Windows).

## Run

- `start_bot.bat` runs the bot non-stop. It checks X every 2 minutes.
- `python bot.py --dry` makes videos into `out/` without posting them. Do this first to check it.
- `python bot.py --url https://x.com/user/status/123` makes one specific post.

To have it start when you log in, add `start_bot.bat` to Task Scheduler with the trigger "At log on".

## Settings (config.json)

| setting | meaning |
|---|---|
| `min_likes` | likes a post needs before it gets clipped (100). Lower = faster but more junk |
| `max_age_hours` | ignore posts older than this |
| `watch_accounts` | X accounts you trust. Their clips are taken at once, with no likes needed |
| `min_minutes_between_posts` / `max_posts_per_day` | posting pace (15 min / 12 a day) |
| `post_mode` | `browser` posts, `none` only makes videos |
| `browser_headless` | `true` hides the browser window, but TikTok blocks hidden browsers more often |

Logs are in `logs/bot.log`, and everything it has seen is in `clips.db`.
