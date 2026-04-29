Our goal is to answer these questions:


* How does Deal or No Deal work with CFR, EFR and MCCFR?
* What does the convergence look like for different algorithms on different versions of the game?
* What policies do these algorithms converge on?
* How can we effectively converge to a strong equilibrium with a huge game tree like this?


We will start by defining the original game implementation, along with some other versions of different sizes and types.

Parameters are defined as: (max_turns, num_item_types, max_single_item_utility)

| Implementation Name | Property    | Parameters |
| ------------------- | ----------- | ---------- |
| Original-Full       | general sum | (10,3,10)  |
| Original-Mini       | general sum | (3,1,1)    |
| Zero-Sum-Full       | zero sum    | (10,3,10)  |
| Zero-Sum-Mini       | zero sum    | (3,1,1)    |

We will use the zero sum versions to do initial tests to make sure that the game implementation works, and make sure that it works with vanilla CFR.

More specifically, the tests we will run are as follows:

| Implementation Name | Algorithm      | Result Metric  |
| ------------------- | -------------- | -------------- |
| Zero-Sum-Mini       | CFR            | Exploitability |
| Zero-Sum-Mini       | EFR (tips)     | Exploitability |
| Zero-Sum-Mini       | EFR (causal)   | Exploitability |
| Original-Mini       | CFR            | Regret         |
| Original-Mini       | EFR (tips)     | Regret         |
| Original-Mini       | EFR (causal)   | Regret         |
| Original-Full       | MCCFR          | Regret         |
| Original-Full       | MCEFR (tips)   | Regret         |
| Original-Full       | MCEFR (causal) | Regret         |
