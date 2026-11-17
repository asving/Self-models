# Blind predictions for the pure-RL runs (written 2026-11-17 before launching; not edited after)

- c (LAM 0, ENT 0.05)
  - Prediction: it still learns to track the controlled stream, because nothing that drove run b's
    tracking depended on the prediction loss.
  - Reward ≥ run b's 1.39, perhaps about 1.43. The head is now free to push token 0 at almost
    probability 1 on the true stream (run b: .88).
  - The belief read-out R² falls below b's .948. Without the prediction loss, the forecasts for the
    uncontrolled streams can drift, and identifying the controlled stream depends on them.
  - Head P(token 0) on the uncontrolled streams drifts away from the untouched forecast of .334.
- d (LAM 0, ENT 0)
  - Prediction: it collapses like run a, with all effort on one fixed stream and reward around 1.1–1.2.
