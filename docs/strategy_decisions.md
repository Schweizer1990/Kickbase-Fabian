# Auction pricing and replacement plans

Every new daily report contains `strategy.decision_plan`. This layer only makes
recommendations; it never places bids, sells players or changes the lineup.

## Auction-aware prices

`auction_bids` counts scheduled market-value updates strictly after the final live
snapshot and before auction expiry. The first future update uses the existing
daily model; each subsequent update applies the configurable `growth_discount`.
This is a scenario rather than a separately trained multi-day forecast.

The suggested bid applies the league segment's historical 75th-percentile
winning premium to that projected expiry value (median fallback). The maximum
bid retains half of the discounted growth scenario over the configured holding
period as a profit buffer. These prices do not imply a probability of winning.
Existing `market_ranking`, `win_ranking` and `bid_guardrails` bid fields are updated
to this price basis. Historical shadow competitor bids remain labelled as
current-market-value comparisons, not observations of competing bids.

Unknown/expired auctions, missing predictions, market values that no longer
match the prediction baseline, manager listings and full club quotas block new
bid recommendations. Refresh after the market-value update; do not add an update
twice. The expected update time is configurable and is not proof an update arrived.

## Replacement decisions

`replacement_comparisons` compares purchasing a target with keeping a profitable
owned trader. It subtracts the acquisition premium and the owned player's growth
forgone during both the auction wait and subsequent holding period. Completed
purchase prices and expected points must be available. Protected core players
and players above the configured points threshold are excluded from sales.
Candidates exceeding the allowed expected-points loss are excluded.

`recommended_replacements` selects up to two distinct sale/buy pairs, checks
cash including existing bids, and simulates their effect on squad and club limits.
Expected points are only a heuristic; a pair is not proof the next starting XI is
legal or better. Clubs already at their limit are excluded from new target
selection rather than automatically selling their point players.

## Bid sequencing

The report simulates existing bids, proposed bids without sales, and the complete
recommended plan. It assumes every bid wins and all sales finish before the
first auction. At 15 owned players, later pending bids are marked cancelled.
The club limit is checked on each assumed acquisition. Unknown expiry/club data
or tied expiries make the plan unsafe rather than claiming a definite sequence.
Recommendations are never instructions to withdraw existing bids.

League rules and scenario parameters live in `strategy_config.json`. Current
defaults: squad 15, club 2, update 22:00 Europe/Zurich, three holding updates,
future growth discount 0.75, at most two replacements, expected-points loss 15,
points-core threshold 60. Ginter, Guirassy, Kobel and Rohr are explicitly protected.

## Verification

Run `python -m unittest discover -s tests -v`. Tests cover update timing,
auction premiums, cancellation of a later bid at 15, two-sale preservation of
three planned purchases, club quotas, manager listings, stale market values,
missing expiry/data, cash and point-core protection. CI executes the same tests.
