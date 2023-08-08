# Q-learning for LMDPs
This repository contains the files used for the experiments performed in the Bachelor Degree Thesis by Cristina Gálvez at UPF.

### Abstract
This thesis investigates linearly-solvable Markov decision processes (LMDPs), a simplified framework derived from Markov decision processes (MDPs) in Reinforcement Learning. LMDPs aim to maximize rewards in sequential decision-making environments, providing a versatile modeling approach.

A common technique employed in MDPs is Q-learning, which approximates expected rewards for state-action pairs iteratively. In LMDPs, Z-learning assigns approximate rewards to states but requires complex modeling.

This thesis proposes the adoption of Q-learning for LMDPs, which closely resembles the structure of the MDP algorithm, computing values for each state-action pair. The newly introduced technique is evaluated in various environments, including small grid-worlds (Frozen Lake) and more complex scenarios (Sokoban). The results highlight the advantages of this method, including its fast convergence. 

### Included in this Repository
This repository includes:
* The original document of the thesis *Q-learning for LMDPs*
* The power-point presentation used in the defense
* A folder called *Frozen Lake* with a python file (with the logic of the game) and a jupyter notebook (with examples of the training process for different methods and their comparisson)
* A folder called *Sokoban* with a python file (with the logic of the game) and two jupyter notebooks (one with examples of the training process for different methods and their comparisson and another one with the code needed for timing the execution of the training processes)

---
_All this work belongs to Cristina Gálvez as part of the thesis Q-learning for LMDPs at UPF_
