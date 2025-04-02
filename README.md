# About
This is a project including an implementation of table-based Q-Learning to play the multi-armed bandit game and a gridworld game.<br>
I did not create the ```Bandit_Sim``` or ```cs7313-mdp``` directories; I only cloned them for use in this project.

### Project Notes
You need to fix Wumpus.<br>
- clip between 1 and 10 should be between 0 and 9.<br>
- terminal states p() should return probability of 1 of going to a terminal state rather than nonzero probs to go to several non-terminal states.<br>
- 4x3 world does not have a wall in it like the book does.<br>
- move probs 0.7, 0.15, 0.15 also do not match the book's 0.8, 0.1, 0.1.<br>

Please adjust the code accordingly.<br>