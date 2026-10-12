# Actual matchday points

`reports/latest.json` now includes `own_matchday_points`, fetched from the
authenticated manager's Kickbase performance endpoint. `matchdays` preserves
the manager's awarded points (`mdp`) for each matchday (`day`), including zero
and negative values. `total_points` is the API season total. The newest labelled
season is selected independently of array order; ambiguous seasons are reported
as unavailable.

`current_matchday` reflects the API `cur` marker, which can refer to an upcoming
matchday. `latest_played_matchday` uses completed, dated player performances as
an anchor; it does not assert that every fixture in that matchday is finished.
`latest_recorded_matchday` also includes a later nonzero current result. Live
scoring and later corrections can change these values. Future zero placeholders
are not used as the latest result; missing scores remain null.

Each `points_profiles` entry also contains `matchday_history`: player points,
minutes, matchday, date, season, competition and the upstream status markers.
These are captured from the already fetched performance response, without an
additional request per player. Both manager results and player histories are
retained in `reports/history.json` snapshots. Older snapshots are not invented
or backfilled from the current squad.

Manager results are authoritative for fantasy scoring. Player histories describe
individual football performances, not historical lineup membership. A sold
player can have contributed to a manager result; a current reserve can have
scored points that did not count for that manager. The report explicitly marks
historical lineup membership as unavailable rather than deriving a false team
total from today's squad. Forecasts (`expected_points_next`) remain separate.

Optional scoring failures produce an unavailable status and do not stop trading
reports. The first run after this change populates the new fields.

API references:
- https://share.apidog.com/bca1f84a-99d7-4f8f-96a5-5e084ee24fe3/get-v4leaguesleagueidmanagersmanageridperformance-29817118e0
- https://share.apidog.com/bca1f84a-99d7-4f8f-96a5-5e084ee24fe3/get-v4leaguesleagueidplayersplayeridperformance-29817126e0

Validation: `python -m unittest discover -s tests -v`.
