# Strategy: <name>

One line: what this strategy bets on.

## State used
Core views and files it reads; what it keeps in `RUN/search/<name>/state.json` (or "none").

## Actions used
Subset of: mutate, fresh, combine, measure, investigate, ship, custom(<role>), stop.

## Parent selection
How a parent (or parents for combine) is chosen from the archive, and the tie-break.

## Round composition
How many actions of each kind per round, scaled to GPUs and eval cost.

## Triggers
Conditions that switch behaviour (plateau, gap size, island stagnation, …), each tied to a state signal.

## Stop
Extra stop conditions beyond the shell end rule (or "shell rule only").

## Tags
Tags executors must set on new nodes (e.g. `island`, `cell`, `approach`), with allowed values.
