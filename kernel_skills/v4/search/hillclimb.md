# Search template: hillclimb (the v3 behaviour; the control arm)

**State used:** the shipped node, the roofline gap per regime, and the closed levers.

**Each round:**
- **Parents:** the shipped node for every mutation. An older node only when the shipped line closed a
  door that it still has open (write which).
- **Mix:** one mutation per cut that holds a large weighted gap to the cap; at most 2–3 cuts per round.
  Cut by regime / size band / cost component.
- **Crossover:** a `merge` of this round's winners on disjoint cuts, at round end.
- **Fresh:** after a 2-round plateau with one design family and a ≥ 5% gap; at once if less than half the
  time is left.
- **Restart:** none; the line only moves forward.

**Good for:** short runs, kernels near their roof, a single dominant regime.
**Weak at:** design changes whose payoff shows up only a round later (a new structure starts slower than
the tuned old one), and regimes that pull the design in different directions.
