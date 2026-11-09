# PiG_Bot
A Human-Like StarCraft II Training Opponent

Built using [python-sc2](https://burnysc2.github.io/python-sc2/index.html) and the [ARES framework](https://aressc2.github.io/ares-sc2/api_reference/index.html).

## Project Overview

**What Inspired This Project?**  
The default StarCraft II in-game AI makes a terrible training partner. This project aims to provide a more challenging, human-like opponent for competitive players.

**Is This For You?**  
- StarCraft II players seeking a challenging AI at different skill levels.

**Why You'll Love This Bot**  
- The bot plays in a way that mimics human opponents.
- Difficulty is adjustable.
- The bot adapts to its opponent’s strategies.

**How You Can Use This Bot**  
- Play against the bot in custom games.
- Use it as a training partner to improve your skills.

**Key Features**  
- Builds according to the Bronze to GM macro standard.
- Scouts and gathers enemy information.
- Reacts to common cheese (Zergling Rush, Cannon Rush, Probe Rush).
- Uses disruptors and basic micro (a-move, stutter step).
- Responds to threats with appropriate unit counters.
- Maintains 3 bases and 66 probes.
- Switches builds based on scouting (pre-defined build switches).
- Fights only when favored.
- Runs 2 build orders per matchup:
    - **PvP:** 4 Gate All-in, 2 Gate Expand
    - **PvZ:** Standard Robo Opener, Double Stargate Phoenix (vs Mutas)
    - **PvT:** Standard Macro Build, Proxy Rax Response

**Tracked Data**  
- Opponent race, map name, game result, build order used, build loss flag, time build switched, enemy openings and units seen, air tech detection, fight outcomes, units lost, expansions taken, and more (see code for full list).

📋 **Community & Task List:** Join the discussion and see current tasks on the [VersusAI Community Forum](https://community.versusai.net/t/pig-bot/49)

## Getting Started

### Prerequisites
- Python 3.11–3.12
- [StarCraft II](https://starcraft2.blizzard.com/en-us/) (free edition works fine)
- Poetry package manager

### Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/Vers-AI/SC2_PiGBot.git
```

2. Navigate to the root folder:
```bash
cd SC2_PiGBot
```

3. Initialize submodules (if you didn't use `--recursive`):
```bash
git submodule update --init --recursive
```

4. Install dependencies using Poetry:
```bash
poetry install
```

5. Configure StarCraft II path (if needed):
   - If you have a non-standard StarCraft 2 installation or are using Linux, adjust `MAPS_PATH` in `run.py`

### Running the Bot

```bash
poetry run python run.py
```

### Linting & Formatting

```bash
poetry run black .
poetry run isort .
```

### Testing

```bash
poetry run python -m pytest tests/
```

CI runs the pytest suite on every PR into `development` and on pushes to `development`/`master` (`.github/workflows/ci.yml`).

## Project Structure
```
bot/
  bot.py                    # Main AresBot subclass; wires all modules together
  constants.py              # All tunable numeric constants (squad radii, thresholds, etc.)
  combat/
    __init__.py             # Re-exports public API
    combat.py               # Core army control logic (squads, roles, engagement decisions)
    unit_micro.py           # Per-unit micro: ranged, melee, disruptor, HT, sentry
    formation.py            # Formation geometry helpers
    scoring_table.py        # Scoring table + priority target selection
    force_field.py          # Sentry force field logic
    group_chase.py          # Group chase behavior
    group_snipe.py          # Group snipe behavior
  managers/
    macro.py                # Economy, production, build order execution
    composition_scoring.py  # Scored composition competition: candidates + considerations
    reactions.py            # Threat detection, assess_threat(), cheese reactions
    scouting.py             # Scout management and intel gathering
    structure_manager.py    # Chrono, recharge, mass recall, building management
  models/
    strategy_belief_model.pkl # BN strategy classification model (4 categories)
  utilities/
    intel.py                # Enemy intel tracking, choke grid creation
    reasoner.py             # Utility-AI reasoner: curves, appraisals, IntelContext
    debug.py                # Visual debug overlays
    nova_manager.py         # Disruptor nova tracking
    use_disruptor_nova.py   # Disruptor nova behavior
    natural_wall_manager.py # Natural wall placement logic
    performance_monitor.py  # Frame-time profiling
    game_report.py          # End-of-game reporting
ares-sc2/                   # Git submodule — do NOT modify src/
config.yml                  # Runtime config (feature flags, build selection)
data/                       # Match replays, memory files, telemetry
tests/                      # Test bots (protoss, terran, zerg)
run.py                      # Entry point to run the bot
```
## Contributing Guidelines

### How to Contribute
1. Check the [community forum thread](https://community.versusai.net/t/pig-bot/49) for current tasks
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide for Python code
- Write meaningful commit messages
- Add comments that explain **why**, not **what** — the code shows the what
- Always comment magic numbers and non-obvious thresholds
- One-liner comments preferred; block comments only when the reasoning is complex
- No comments that just restate the code
- Update documentation when changing functionality

## Resources

- [python-sc2 Documentation](https://burnysc2.github.io/python-sc2/index.html)
- [ARES Framework Documentation](https://aressc2.github.io/ares-sc2/api_reference/index.html)

## License

This project is released under the [MIT License](LICENSE).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

Want to contribute and be recognized? We invite you to join our AI Makers community: [https://versusai.net/join/](https://versusai.net/join/)

Community members get access to contribution opportunities, recognition, and more!

This project follows the [all-contributors](https://allcontributors.org) specification. Contributions of any kind welcome!
