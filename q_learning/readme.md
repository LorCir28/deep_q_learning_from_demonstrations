# Q-learning

Implementation of Q-learning via DQN with buffer replay and target network. The algorithm was tested on the main project environment (Montezumas's Revenge) and on another environment (Pong). This dual approach is intended to demonstrate that Q-learning, in sparse reward environments such as Montezumas, fails miserably.

## Directory - Models

Contains the towed models on the two environments.

- __mz_net, mz_tgt__: models of the core network and target network for Montezuma

- __pong_net, pong_tgt__: models of the main network and the target network for Pong


## Directory - Notebooks

Contains the Colab notebook (working) on which the nets were trained.

## Directory - Scripts

It contains the exploded Colab file in all its main classes (excluding the __play.py__ file which was written separately):

- __agent.py__ : the file in which all the classes are integrated to realize Q-learning
- __buffer.py__ : realizes the buffer replay
- __dqn.py__ : neural network structure for the main network and the target network
- __play.py__ : tests the operation of the algorithm
- __wrap.py__ : various regularizations and normalizations taken from OpenAI in order to improve the performance of the networks

## Training
Run the notebook file on Google Colab

## Testing
__Montezuma:__
- Change the name of the environment selected on the __play.py__ file by uncommenting the constant __DEFAULT_ENV_NAME = "MontezumaRevenge-v4"__
- Run the command __python ./play.py -m ../models/mz_net.pt__ from the terminal.

__Pong:__
- Change the name of the environment selected on the __play.py__ file by uncommenting the constant __DEFAULT_ENV_NAME = "Pong-v4"__
- Run the command __python ./play.py -m ../models/pong_net.pt__ from the terminal.