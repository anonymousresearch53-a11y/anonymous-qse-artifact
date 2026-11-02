"""Minimal, experiment-friendly two-step chain RL package.

This package keeps the environment logic separate from:
1. dataset preparation,
2. reward construction, and
3. model training.

The intent is to make it easy to swap rewards, hyperparameters,
or algorithms without having to edit one giant training notebook.
"""
