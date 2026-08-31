# core/ — shared library
- model.py: GPT / TTTNet / Block architectures (return_hidden, hooks).
- factors.py: Mess3 + asym3 factor adapters over ~/comp_icl's generator
  (CompositionMixture, belief_filter). Importers add ~/comp_icl to sys.path
  themselves.
- whitebox.py: exact-recompute + synthetic-program kit (house whitebox style).
- probes.py: ridge probes, subspace tools, readouts.
Experiment clusters symlink what they need from here; edit in place, all
clusters see it.
