![alt text](https://img.shields.io/badge/Course-Udacity--DRL-blue.svg) 

# **Project 2: Continuous Control** 

The goal of this project is to teach an autonomous agent with a double-jointed arm to keep its hand in a moving target location for as long as possible. 

![JointedArmAgents](./report_images/jointed-arm-agents.png "JointedArmAgents")

It is a part of the Udacity nanodegree Deep Reinforcement Learning. 

---

### Problem Setup

##### Environment
The environment consists of a double-jointed arm and a moving target location. There is also a correpsonding training environment where the agent have 20 parallel workers to collect experiences faster.

##### Rewards
The reward function is:   
`0` for moving the arm.  
`+0.1` for each timestep the hand is kept in the moving target location.

##### Goal
The goal of the agent is to maximize episodic reward. The task is considered solved if the average reward over 100 episodes is at least `+30`.

##### Observation Space
The agent observes the environment using a sensor yielding a `33`-dimensional vector that includes position, rotation, velocity, and angular velocities of the arm. Thus, the observation space is continuous.

##### Action Space
To control the arm the agent applies torques to each joint, resulting in a `4`-dimensional vector. Thus, the action space is also continuous.

---

### The Double-Jointed Arm Agent

##### Implementation
The double-jointed arm agent is implemented and trained in the Jupyter notebook `Continuous_Control.ipynb`.  
The weights of a trained agent are stored in `double_jointed_arm.pth`.

##### Description
The details of the double-jointed arm agent can be found in `Report.md`. 

---

### Getting Started
To train the agent described in the notebook, follow this instruction (MacOS):

1. Create a new virtual environment running Python 3.6:
```
$ conda create -n <my_env_name> python==3.6 
```

2. Activate virtual environment:
```
$ source activate <my_env_name>
```

3. Install dependencies (use specific version combo below):
```
$ conda install scipy
$ pip install tensorflow==1.7.1
$ pip install torch==0.4.0
$ pip install mlagents=0.4.0
```

4. Download DoubleJointedArm environments and unzip them:  
    * [training environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)  
    * [evaluation environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)

5. Start Jupyter Notebook server:
```
$ jupyter notebook
```

6. Run notebook `Continuous_Control.ipynb`