# Alpha Nagibator

Implementation of self-play based reinforcement learning for Checkers (Russian variant) based on the AlphaGo Zero methods. 

The implementation uses Python and Keras. It is based on [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) framework so you can easily implement other variants of checkers. 

To play against the trained model:
```
python checkers_pit.py
```

If you plan to train your model or improve the bundled one look at checkers_main.py

### How to play
To play against model you need Python and Keras. GPU is strongly recommended to save your time. The code is compatible both with Python 2.7 and Python 3.6. The only reason to use Python 2.7 was AMD GPU support at the early 2018. Since April 2018 you can use Python 3 with AMD and NVidia GPUs. The code was tested on Ubuntu 16.04 and Windows 7/10.

When you run ```python checkers_pit.py``` you will see a 8x8 board in pseudo-graphics, short statistics and a list of possible moves. 
![alt tag](https://github.com/evg-tyurin/alpha-nagibator/raw/master/help/how-to-play.png)

Then you type a move and wait for the response. Then you type your next move and so on. If your move is a capture of more than one opponent's pieces then you type the first capture, press enter, wait for the response, then type the next capture etc.

### Pretrained model
I trained a Keras model (51 iteration, 50-400 episodes per iteration and 50-100 MCTS simulations per turn). The pretrained model can be found in ```pretrained_models/checkers/keras/```. Below is the performance of the model against the 20th iteration of the same model with the number of iterations.
![alt tag](https://github.com/evg-tyurin/alpha-nagibator/raw/master/pretrained_models/checkers/keras/rev250-1i-45i.png)

The performance of the model was estimated as a total result in 100-games match against reference player. Reference player is the 20th iteration of the same model.

If you want to check the performance against strong classical engine you can use ```engine_server.py``` and any Kallisto_4 compatible engine. Spoiler: classical engine will beat the model at 100-0.

### Thanks to 
* [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
* [chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)
* [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [progress](https://github.com/verigak/progress).
