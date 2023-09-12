# Lunar example

Set of tests for using lunar lander as gym (gymnasium) example.

This code allows later to compare with our custom environments, so they behaves as expected in the general gym environments.
This checks also prevents using shiny custom datatypes or non-standard interactions that confuse people importing the custom environment or break general RL algorithms that cannot be applied if conditions are not the same.

## Lunar with SB3

Note that we have started from a sample for cartPole and just changing by lunar, because even different problems, they share similar features like discrete actions and discrete states, optimal solution is clear, etc, so this will be the same for our SecurityEnvironments (with more reward types) where maybe not optimal but at least good solutions should be learned with few iterations and convergence should be very fast as well.
