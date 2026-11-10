# Counter Table Research — Liquipedia (LotV pages), pulled 2026-10-10; coverage pass pulled 2026-11-06
Source API: liquipedia.net/starcraft2/api.php (gzip required), pages: <Unit>_(Legacy_of_the_Void).
Citations reference the same "# LP:" format as the existing SCORING_TABLE in bot/combat/scoring_table.py.

## 2026-11-06 coverage pass — what it pulled
Task 2 (plan/utility_refactor_plan.md) audits attacker/target coverage of SCORING_TABLE. Gap analysis:
every Protoss unit we build already had ATTACKER rows; the gaps were on the TARGET side — our own unit
types appearing as the enemy (mirror matchups, i.e. PvP): Dark Templar, Disruptor, High Templar, Archon
and Sentry had 0 rows, Immortal and Colossus had 2 each. Pages pulled for this pass (LotV wikitext via
the API above): Dark Templar, Disruptor, High Templar, Archon, Sentry, Immortal, Colossus. New pairs
below are annotated with the quote they rest on.

Row semantics used throughout (matches the existing table): (OUR_TYPE, ENEMY_TYPE) is positive when our
unit type is the better answer to that enemy type, negative when that enemy type counters us (e.g. the
existing (COLOSSUS, VIKINGFIGHTER) -10.0 — "Vikings...primary counter to Colossi").

## ADEPT (Gateway, Light, Biological; built today by PiG_Bot)
- LP: "strong against early Zealots and other light units"
- LP: "Adepts have no anti-air attack" (Muta/mass air harass kills them)
- LP: PvZ: Roaches out-tank; snowball vs bio in PvT "when paired with Phoenix"
- Pairs: (ADEPT, MARINE) +6.0 [light, early numbers]; (ADEPT, ZERGLING) +6.0 [LP: strong vs light/Zealots]; (ADEPT, ROACH) -6.0 [armored, out-tanks]; (ADEPT, MUTALISK) -6.0 [no anti-air]

## SENTRY (Gateway, Light, Mechanical; built today)
- LP: Forcefields "guard the ramp and force favorable engagements" (alternative defense vs Zealots)
- LP (Guardian Shield): "effective against any low damage, high rate of attack type units (such as Marines)"
- Pairs: (SENTRY, MARINE) +3.0 [GS reduces marine damage; support role]; (SENTRY, ZERGLING) +3.0; (SENTRY, ZEALOT) +4.0 [LP: FF defense vs Zealots]
- NOTE: support/utility unit; scores stay small (it enables, it doesn't kill)
- SITUATIONAL (research 2026-10-12): Sentry value is a FIXED count, not a scaling counter.
  LP Sentry page: "a critical mass of six sentries regenerate energy quickly enough to keep
  a sufficiently small ramp or choke continually blocked with Force Fields" — FF utility
  saturates at ~6; learning-sc2: "a minimum of eight Sentries is recommended against Zerg"
  when moving out. Pro guidance = situational investment (~6-10), independent of enemy
  army size. Base compositions already encode these ratios (e.g. Soul Train 0.32 proportion
  from "10 Zealots, 6 Sentries, 3 Immortals"). Nudging must NOT stack per-enemy bonuses on
  top → rows are in SITUATIONAL_ROWS; production nudging excludes them (macro.py
  _compute_effectiveness), combat targeting keeps them.
- 2026-11-06 (mirror pass — Sentry as the ENEMY target):
  - LP: Guardian Shield "makes this a high priority target for opposing armies" (kill the Sentry,
    deny the GS/FF)
  - LP (in-game help infobox): Sentry is weak vs Stalker, strong vs Zealot/Zergling
  - LP (High Templar page, Feedback table): the Sentry is one of the units "susceptible to Feedback"
  - LP (Archon page): "Sentries are worthless because Archons break force fields"
  - Pairs: (STALKER, SENTRY) +6.0 [high priority target]; (HIGHTEMPLAR, SENTRY) +6.0 [Feedback];
    (ARCHON, SENTRY) +4.0 [FF useless vs Archon]
  - CORRECTION (2026-11-06): the table's Sentry section header says "no direct attack in LotV" — the
    LotV Sentry page says otherwise: "In addition to their spells, Sentries have a primary attack that
    is capable of hitting both air and ground targets, although their damage output is rather weak"
    (Disruption Beam, 6 dmg, +4 vs shields, range 5). The existing Sentry rows stay as they are — their
    values encode GS/FF support, not the beam.

## DARK TEMPLAR (Dark Shrine, Light, Psionic, Biological)
- LP: "even a small number of DT can decimate a large number of workers very quickly" if undetected
- Pairs: (DARKTEMPLAR, MARINE) +5.0 [45-dmg melee shreds light if no detection]
- NOTE: detection-dependent; worker-line value not modeled in this table (workers excluded)
- 2026-11-06 (mirror pass — Dark Templar as the ENEMY target; PvP):
  - LP: "Although they are fragile when detected, Dark Templars deal huge burst damage" (40 HP + 80
    shield, 45 dmg melee — dies fast to anything ranged once an Observer spots it)
  - LP: "The DT rush is normally ended when the opponent's first Observer comes into play"
  - LP: "DTs are powerful units, and can repel small numbers of Zealots even if they are visible"
  - LP (Archon page): "Players often merge Archons... if the opponent has detectors that vitiate the
    Dark Templars' main strength" (detection removes the DT's edge — the merge form then wins)
  - LP (Disruptor page): the Purification Nova "can kill the following units with one shot: ... High
    Templars, Dark Templars, Adepts, Disruptors, and Sentries" and "Disruptors also allow one to kill
    stealth units without detection"
  - Pairs: (DISRUPTOR, DARKTEMPLAR) +10.0 [Nova one-shot + kills stealth without detection];
    (STALKER, DARKTEMPLAR) +8.0 [fragile once detected — Observer + range];
    (IMMORTAL, DARKTEMPLAR) +8.0 [45-dmg melee vs 550 eHP ranged];
    (ARCHON, DARKTEMPLAR) +6.0 [detection vitiates the DT];
    (ZEALOT, DARKTEMPLAR) -8.0 [DTs repel Zealots]
  - Mirror (our DT vs their Disruptor): (DARKTEMPLAR, DISRUPTOR) -8.0 [Nova one-shots Dark Templars]

## DISRUPTOR (Robotics Facility, Armored, Mechanical; built today)
- LP: no auto attack — damage comes from the Purification Nova, which "hits friendly and enemy units
  alike (except friendly Disruptors)"
- LP: "The Disruptor's Purification Nova can kill the following units with one shot: Marines, Hellions,
  Reapers, Widow Mines, Zerglings, Banelings, Hydralisks, Infestors, Zealots, Stalkers, High Templars,
  Dark Templars, Adepts, Disruptors, and Sentries."
- LP: "To kill Marauders, Ghosts, Hellbats, Cyclones, Siege Tanks, Vikings in Assault mode, Queens,
  Roaches, Ravagers, Swarm Hosts, or Lurkers, two Novas are needed."
- LP: "Three Novas are needed to kill an Immortal with the first shot being absorbed by the Immortal's
  barrier." / "The Disruptor needs three or more Novas to take out Thors (4), Ultralisks (5), Colossi (3),
  and Archons (4)."
- LP: "The Disruptor is very good against ground units that cannot move, such as Siege Tanks in siege
  mode and Lurkers, or even enemy Disruptors."
- Pairs (attacker side, added 2026-11-06): (DISRUPTOR, ZEALOT) +10.0 [Nova one-shot];
  (DISRUPTOR, ADEPT) +8.0 [Nova one-shot]; (DISRUPTOR, HIGHTEMPLAR) +10.0 [Nova one-shot];
  (DISRUPTOR, DARKTEMPLAR) +10.0 [Nova one-shot]; (DISRUPTOR, COLOSSUS) -4.0 [3 Novas to take out a Colossus]
- 2026-11-06 (mirror pass — Disruptor as the ENEMY target):
  - LP (PvP): "Immortals are sometimes used to handle Disruptors numbers"
  - LP (PvP): "Stalkers can use Blink to avoid damage from the Nova"
  - LP (PvP): "Phoenix are often used to disable Disruptors once the Nova has been launched, which
    prevents all damage"
  - LP (Archon page): "In terms of splash counters however Archons are countered by Disruptors due to
    the latter's bonus vs shields and the former's inability to reliably dodge purification novas"
  - Pairs: (IMMORTAL, DISRUPTOR) +8.0 [Immortals handle Disruptor numbers; 3 Novas to kill one];
    (STALKER, DISRUPTOR) +6.0 [Blink dodges the Nova, the Disruptor has no auto attack];
    (PHOENIX, DISRUPTOR) +8.0 [Phoenix disable the Disruptor once the Nova is launched];
    (ARCHON, DISRUPTOR) -8.0 [Disruptors counter Archons];
    (ZEALOT, DISRUPTOR) -8.0 [Nova one-shots Zealots]
  - Mirror (our Disruptor vs their Archon): (DISRUPTOR, ARCHON) -8.0 [4 Novas to take out an Archon]

## HIGH TEMPLAR (Gateway, Light, Biological, Psionic; built today)
- LP: "Although it has an almost harmless attack and is relatively weak in terms of hit points and
  shields, the very powerful Psionic Storm can be vital for the success of Protoss."
- LP (in-game help infobox): High Templar is strong vs Marine/Sentry/Hydralisk, weak vs
  Ghost/Colossus/Roach
- LP (Feedback): "Used to snipe enemy spell casters like Ghosts and Infestors... Dealing pure damage,
  ignores all types of armor." Susceptible units include (Protoss column) High Templar, Sentry,
  Phoenix, Oracle
- LP: "High Templars can also be used to Feedback Phoenixes and Oracles, thus being a good defensive unit
  as well as an offensive one."
- 2026-11-06 (mirror pass — High Templar as the ENEMY target):
  - Pairs: (DISRUPTOR, HIGHTEMPLAR) +10.0 [Nova one-shot];
    (COLOSSUS, HIGHTEMPLAR) +8.0 [in-game help: HT is weak vs Colossus];
    (STALKER, HIGHTEMPLAR) +6.0 [almost harmless attack, weak HP/shields];
    (ARCHON, HIGHTEMPLAR) +6.0 ["Archons are strong against almost every Protoss unit"]
  - Attacker side (Feedback): (HIGHTEMPLAR, PHOENIX) +8.0 [Feedback drains and damages];
    (HIGHTEMPLAR, ORACLE) +8.0 [Feedback]; (HIGHTEMPLAR, SENTRY) +6.0 [Feedback]
  - Mirror (our HT vs their Colossus): (HIGHTEMPLAR, COLOSSUS) -6.0 [in-game help: HT is weak vs Colossus]

## ARCHON (Templar Archives merge, Massive, Psionic; built today — HT/DT merge)
- LP: "Archons are strong against almost every Protoss unit, but most notably when up against Gateway
  units. Zealots are eaten up by splash and bonus damage while Adepts can't deal any real damage
  against them. Sentries are worthless because Archons break force fields."
- LP: "Generally the way to counter Archons is by simply focus firing them." / "Archons' tremendous
  shield points can withstand terrible punishment, offsetting their exceptionally low health: an Archon
  has only ten health points beneath its shield."
- LP: "In terms of splash counters however Archons are countered by Disruptors due to the latter's bonus
  vs shields and the former's inability to reliably dodge purification novas without prism pickup micro."
- LP: "Against Mech in LOTV, the Archon is now the most durable unit against Siege Tank fire, due to its
  lack of an armored tag and changes to the Immortal's shield ability."
- LP: "Their bonus against biological units makes Archons effective against all Zerg units."
- LP: "Another use for Archons in PvP is against Void Rays and Carriers. The former being susceptible to
  clumping and having poor damage against Archons. While Carriers do have a better fighting chance,
  Archons with shield upgrades, sufficient numbers and under shield batteries can beat Carriers by
  taking down their interceptor complements rendering the ships useless."
- 2026-11-06 (mirror pass — Archon as the ENEMY target):
  - Pairs: (STALKER, ARCHON) +6.0 [focus fire; only 10 HP under the shields];
    (IMMORTAL, ARCHON) +5.0 [sustained DPS strips the shields, then the 10 HP];
    (DISRUPTOR, ARCHON) -8.0 [4 Novas to take out an Archon];
    (HIGHTEMPLAR, ARCHON) -6.0 ["Archons are strong against almost every Protoss unit"];
    (VOIDRAY, ARCHON) -6.0 [Archons are the answer to Void Rays];
    (CARRIER, ARCHON) -6.0 [Archons can beat Carriers]
  - Attacker side: (ARCHON, SIEGETANKSIEGED) +8.0 [most durable unit against Siege Tank fire];
    (ARCHON, SIEGETANK) +6.0 [same durability, weaker mobile gun];
    (ARCHON, ROACH) +8.0 [bonus vs biological, no armored tag];
    (ARCHON, ADEPT) +6.0 [Adepts can't deal damage]; (ARCHON, VOIDRAY) +6.0 [poor damage vs Archons];
    (ARCHON, CARRIER) +6.0 [beat Carriers]; (ARCHON, DARKTEMPLAR) +6.0 [detection vitiates the DT];
    (ARCHON, SENTRY) +4.0 [FF useless]; (ARCHON, HIGHTEMPLAR) +6.0 [strong vs Protoss units];
    (ARCHON, DISRUPTOR) -8.0 [Disruptors counter Archons]

## IMMORTAL (Robotics Facility, Armored, Mechanical; built today)
- LP (PvP): "Immortals deal a large amount of damage to Stalkers and Colossi, but are quite ineffective
  against enemy Zealots."
- LP (PvP): "if Extended Thermal Lances is researched, the player with 9-range Colossi retains a large
  range advantage over the 6-range Immortals and often can position his Colossi to avoid Immortal fire
  while still attacking"
- LP (PvT): "Immortals are incredibly effective against small numbers of sieged Siege Tanks due to their
  slow rate of fire"
- 2026-11-06 (mirror pass — Immortal as the ENEMY target):
  - Pairs: (COLOSSUS, IMMORTAL) -6.0 [Immortals deal a large amount of damage to Colossi];
    (ZEALOT, IMMORTAL) +6.0 [Immortals are quite ineffective against Zealots]
  - Attacker side: (IMMORTAL, COLOSSUS) +8.0 [Immortals deal a large amount of damage to Colossi]

## COLOSSUS (Robotics Facility, Massive, Mechanical; built today)
- LP (PvT): "They are however very weak against large masses of Marauders, both because Colossi have no
  bonus damage against them and because Marauders deal bonus damage against the Armored Colossi."
- LP (PvZ): "Alongside that, Lurkers rip the standard complementary gateway unit to Colossi, the
  Stalker, which would otherwise normally provide anti-armor and anti-air."
- LP: "it is only outranged by the Terran Siege Tank in its immobile Siege Mode and the Zerg Lurker with
  Seismic Spines"; the Colossus "can kite ground units using the terrain to its advantage"
- LP (PvP): "Colossi are not typically used in PvP matches in Legacy of the Void. Protoss players
  typically use Immortal, Archon, Chargelot (IAC) or Blink/Disruptor compositions instead"
- 2026-11-06 (mirror pass — Colossus as the ENEMY target):
  - Pairs: (DISRUPTOR, COLOSSUS) -4.0 [3 Novas to take out a Colossus];
    (HIGHTEMPLAR, COLOSSUS) -6.0 [in-game help: HT is weak vs Colossus]
  - Attacker side: (COLOSSUS, MARAUDER) -8.0 [very weak against mass Marauders];
    (COLOSSUS, SIEGETANKSIEGED) -4.0 [the only ground unit that out-ranges it];
    (COLOSSUS, IMMORTAL) -6.0; (COLOSSUS, HIGHTEMPLAR) +8.0 [in-game help: HT weak vs Colossus]
  - Same quote, Stalker side: (STALKER, LURKERMPBURROWED) -6.0 [the Lurker strips the Stalker, the
    Colossus comp's anti-armor/anti-air unit]

## PHOENIX (Stargate, Air, Light, Mechanical)
- LP: "great hit-and-run"; "effective at disabling caster units or destroying them outright. Most casters have light armor"; lifts high-threat units (Siege Tanks) — Graviton Beam per LP infobox
- Pairs: (PHOENIX, MUTALISK) +8.0 [classic Muta counter]; (PHOENIX, BANSHEE) +6.0 [light air, melts]; (PHOENIX, MEDIVAC) +6.0 [lifts/heal-kill]; (PHOENIX, VIPER) +6.0 [air, disables]; (PHOENIX, ORACLE) +5.0 [mirror: out-fights caster]; (PHOENIX, VIKINGFIGHTER) -8.0 [armored air, out-ranges 9>5, LP-tier hard counter]

## VOID RAY (Stargate, Air, Armored, Mechanical)
- LP: "versus an opponent that gets a lot of mechanical units like the Thor or the Tank, a few Void Rays can focus down the armored units very fast" (ramp damage)
- LP: drop defense — "if a boosted Medivac passes right by a charged Void Ray it will barely get through"
- LP: "Stalkers can deal with Void Rays easily because of the ease of making large amounts" ; VR vs Tempest: "Void Rays are extremely strong against Tempest (bonus damage against armored)"; PvP post-3.8: Carriers "easily countered by Stalkers + Sentries, Tempests, and Void Rays"
- Pairs: (VOIDRAY, SIEGETANK) +4.0; (VOIDRAY, THOR) +5.0 [LP: focus down armored]; (VOIDRAY, BATTLECRUISER) +8.0 [ramp on massive]; (VOIDRAY, TEMPEST) +8.0 [LP explicit]; (VOIDRAY, CARRIER) +6.0 [LP explicit]; (VOIDRAY, MEDIVAC) +5.0 [drop deterrent]; (VOIDRAY, MARINE) -6.0 [mass small arms]; (VOIDRAY, STALKER) -8.0 [LP explicit mirror]; (VOIDRAY, HYDRALISK) -6.0 [mass ranged]

## TEMPEST (Fleet Beacon, Air, Armored, Massive)
- LP: "Tempests can be used with late game army to snipe enemy Colossi and forcing the enemy to retreat"; "Void Rays are extremely strong against Tempest"
- Pairs: (TEMPEST, COLOSSUS) +8.0 [LP: snipe role, super-long range]; (TEMPEST, CARRIER) +7.0 [range war on big air]; (TEMPEST, BATTLECRUISER) +8.0 [bonus vs massive]; (VOIDRAY counter is a LOSS:) (TEMPEST, VOIDRAY) -6.0 [LP explicit]; (TEMPEST, MUTALISK) -4.0 [swarm bypasses range]

## CARRIER (Fleet Beacon, Air, Armored, Massive)
- LP: "extremely powerful versus stationary defenses and AI controlled units, since the AI will target the individual Interceptors first, not the Carriers themselves" (Patch 3.8.0 made Carriers "the dominant late game unit choice" in PvP)
- LP: Vikings: "killed in 2 volleys by Carriers... less than a third of the cost... the only match-up where the Carrier will lose the resource war"
- LP: PvP counters: "Stalkers + Sentries, Tempests, and Void Rays"; Zerg: Corruptors (same family as Colossus-counter line)
- Pairs: (CARRIER, PHOTONCANNON) +6.0 [vs static D, range outside cannon]; (CARRIER, SPORECRAWLER) +7.0; (CARRIER, SPINECRAWLER) +7.0; (CARRIER, VIKINGFIGHTER) -6.0 [LP resource war]; (CARRIER, CORRUPTOR) -5.0; (CARRIER, VOIDRAY) -5.0 [LP explicit PvP]

## ORACLE (Stargate, Air, Armored, Mechanical, Psionic) — EXCLUDED AS AN ATTACKER BY DESIGN
- LP: "flying harassment and support caster"; PvZ meta harass "forces a response... Spore Crawlers and Queens"; "fairly expensive and fragile"
- Pulsar Beam: ground-only; vs ARMY units it has no matchup advantage — its value is economic harass + Revelation (anti-stealth intel) + Stasis Ward (traps)
- No ATTACKER pairs: consistent with the table's own rule "Only add entries where there's a real matchup advantage."
- 2026-11-06: as an ENEMY target the Oracle keeps its rows — (HIGHTEMPLAR, ORACLE) +8.0 [LP High Templar
  page: the Oracle is "susceptible to Feedback"; "High Templars can also be used to Feedback Phoenixes
  and Oracles"], plus the existing (STALKER, ORACLE) +8.0 and (PHOENIX, ORACLE) +5.0.

## MOTHERSHIP (Nexus ult, Air, Armored, Massive, Psionic) — EXCLUDED FROM TABLE BY DESIGN
- LP: "the primary use of the Mothership is its powerful support abilities" (Cloaking Field, Mass Recall, Time Warp); "Mothership play is usually ineffective against Terran"; niche support in Tempest-vs-BC battles
- No pairs: support ult; not a counter relationship vs any unit.
