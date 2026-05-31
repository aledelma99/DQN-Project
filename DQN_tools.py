import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque
import time
import matplotlib.pyplot as plt
from scipy import stats
import gymnasium as gym
import pandas as pd
from IPython.display import display

# Create a safe fallback device for the file initialization
file_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
  def __init__(self, state_size, action_size, seed = 42):
    super(Network, self).__init__()

    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, action_size)

  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return self.fc3(x)
  

class ReplayMemory(object):

  #Parameters initialization
  def __init__(self, capacity, device=file_device):
    self.device = device 
    self.capacity = capacity
    self.memory = []

  #Method that firstly put inside the experience (tuples of S,A,R,S') and secondly,
  #if the previously defined capacity is exceeded, it eliminates the first element of the buffer (FIFO logic)
  def push(self, event):
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  #Method used to sample randomly the decorrelated past experiences to update the weights of the online network
  #The number of sampled elements depends on the dimension of the batch size previously defined as hyperparameter
  def sample(self, batch_size):
    experiences = random.sample(self.memory, k = batch_size)
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return states, next_states, actions, rewards, dones
  

class Agent():
  def __init__(self, state_size, action_size, alpha,
               target_update, tau,
               target_update_steps, gamma, save_dir="./checkpoints", device=file_device):

    #Parameters initialization
    self.device = device #(CPU or GPU)
    self.state_size = state_size #Dimension of the State space
    self.action_size = action_size #Dimension of the Action space
    self.local_qnetwork = Network(state_size, action_size).to(self.device) #First create the local network, then pass it from RAM to the device (CPU or GPU)
    self.target_qnetwork = Network(state_size, action_size).to(self.device) #Same as local network, creating the target network (updated slowly)
    self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict()) #This loads the target's weights, copying the ones of the online one to obtain the same initial configuration
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = alpha) #Optimizer used to update the weights of the online network, the size of these updates is regulated by the learning rate parameter
    
    
    self.batch_size = 100 #Dimension of the batch
    self.replay_buffer_size = 100000 #Dimension of the memory buffer
    self.maximum_number_timesteps_per_episode = 1000 #Max timesteps per episode
    self.epsilon_starting_value = 1.0 #Starting Value of epsilon during the run
    self.epsilon_ending_value = 0.01 #Ground floor for epsilon

    
    self.save_dir = save_dir #Save directory

    # Here we instantiate the memory buffer previously defined in the class "ReplayMemory" with capacity as "replay_buffer_size"
    self.memory = ReplayMemory(self.replay_buffer_size, self.device) 
    
    self.t_step = 0  #Regulates how often update the networks
    self.learn_step = 0 #Counter used to assess the occurrence of the target's update
    self.gamma = gamma #Parameter used inside the online's update formula, it regulates the importance of future rewards, discounting them
    self.tau = tau #Parameter used inside the soft update formula, it regulates the importance of the difference between target and online weights
    self.target_update = target_update #Type of the target's update rule; it can be either "soft" or "hard"
    self.target_update_steps = target_update_steps #It defines how many update steps are required to update the target using the "hard" method
    self.checkpoint_episode = 100 #It regulates how often the models are stored inside the disk

  #step will be called by the agent after an action is taken, passing the gained experience to the memory buffer
  def step(self, state, action, reward, next_state, done): # The experience is used as input for this method
    self.memory.push((state, action, reward, next_state, done)) #Append the tuple of (S,A,R,S,Done)
    self.t_step = (self.t_step + 1) % 4 #Used to update the model every 4 t-steps of the episode
    loss_value = None #Instantiate the loss value as None before any computation followed by the learn method
    if self.t_step == 0 and len(self.memory.memory) > self.batch_size: #Used to update the model every 4 t-instances, but we have to wait until the memory buffer has enough experience to sample the elements randomly in a proper way (avoiding correlation)
      experiences = self.memory.sample(self.batch_size) #We sample from the memory buffer k random elements to get the past experience to update the online
      loss_value = self.learn(experiences) #Compute the loss function values used for the plots and update the networks using the learn method injecting the sampled experiences
      return loss_value

    return None #If the network is not updated during this t-steps, return nothing and continue the training process

  #act is used by the agent to take an action A given the current state S of the environment
  def act(self, state, epsilon = 0.):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) #First transform the input state from array to tensor, then add to this latter a batch dimension, required to be properly used by the network
    self.local_qnetwork.eval() #Activate the evaluation mode to avoid updates
    with torch.no_grad(): #To avoid the computations of gradients because this is not required here
      action_values = self.local_qnetwork(state) #Compute the Q function for every pair of State and Action 1,2,3... Q(state,a1), Q(state, a2)
    self.local_qnetwork.train() #Activate again the training mode to update the network successively
    max_q_value = action_values.max().item()
    #Exploration VS Exploitation
    if random.random() > epsilon: #If the sampled number is greater than epsilon, exploit acting greedily
      return np.argmax(action_values.cpu().data.numpy()), max_q_value #Take the set of all the Q functions ((s,a1),(s,a2)...), take the index of action A* associated with the highest Q function ---> Q(state,A*); Then convert the tensor to an array
    else: #Explore by taking a random action using the ones inside the Action space
      return random.choice(np.arange(self.action_size)), max_q_value #First create an array composed by the action's index (from 0 to self.action_size -1), then take a random action 

  ##Methods to update the networks' weights
  def learn(self, experiences):

    states, next_states, actions, rewards, dones = experiences
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) #Compute the Q function using the target network, detach remove a tensor from the graph used to compute the gradients successively (because target must be fixed)
                      #max will return a tuple (values, indices), since we want the Q's values, we take [0]. Then, we add a dimension to create a suitable batch for the following calculations, creating a column vector 
    
    q_targets = rewards + (self.gamma * next_q_targets * (1-dones)) # Bellman Equation adapted to the DQN
                                                                    #The final term (1-dones = 1-1 = 0 if done is terminal or trucated) is used to make the whole product 0 if the reward obtained is from a TERMINAL STATE, which is not linked to future states and actions
    
    q_expected = self.local_qnetwork(states).gather(1, actions) #Q function computed using the local network                                                                                                    
                                                                #given the Q function computed, gather is used to take the actions taken in thes sampled states by the "step" method before
    
    loss = F.mse_loss(q_expected, q_targets) #Compute the the mean squared error between the q_targets and expected
    self.learn_step += 1 #Increase the learn step by 1 for the eventual hard_update 

    self.optimizer.zero_grad() #Reset the gradients to 0
    loss.backward() #Now, compute the gradients of the loss with respect to the local_network parameters
    self.optimizer.step() #Adjust the local_network parameters so that the loss function is minimized

    #Now we update the target network
    if self.target_update == "soft":  #Made every time the local_network is updated
      self.soft_update()

    elif self.target_update == "hard":
      if self.learn_step % self.target_update_steps == 0: #We keep the target's weight fixed until this modulo returns 0
        self.hard_update()

    return loss.item() #Return the value of the loss to be used for the next plots

  def soft_update(self): #Update the target network using the soft update rule at every time step inside the episode
    #Bind together the weights coming from both networks
    for target_param, local_param in zip(self.target_qnetwork.parameters(),
                                         self.local_qnetwork.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
      #Update the weights of the target network using the formula : θ′ ← τ θ + (1 −τ )θ′

  #This is the alternative target's update rule, where the weights are updated copying them from the online after its previous update
  def hard_update(self): #Directly copy the weights of the online network
    self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())


def compute_avg_q(model, states, device=file_device): #The function takes as input the local network, the set of fixed states and the device (CPU or GPU)
    model.eval() #Used to avoid any update
    q_values = [] #Empty list, which will store the maximised Q-values computed for each sampled state

    with torch.no_grad(): #Disable gradient calculations for efficiency during inference
        for s in states:
            s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) #Create a tensor for every state in order to be used as input for the neural network
            q = model(s) #Here the Q-values are computed passing them to the local network
            q_values.append(q.max().item()) #For every state, we store the maximised Q-value obtained by acting greedily (A*)

    model.train() #Now that the work is ended, we turn again on the model's train mode
    return np.mean(q_values) #We return V(S) as the mean of all the stored Q-values


#This function is used to save the model, splitting the cases of checkpoint, environment solved or not
def save_model(agent, experiment_name, current_cfg, current_results, episode, solved, run_idx, checkpoint = False):
    save_dir = agent.save_dir  #Set the save_directory using the attribute of the class agent
    
    if not os.path.exists(save_dir): #check the existence of the save_directory, otherwise create it
        os.makedirs(save_dir) 

    #Here che type of the checkpoint will be evaluated, creating different filenames using the experiment name, the number of run (0,1,2,3,4,5), the checkpoint episode and if the environment was solved or not
    if checkpoint:
        filename_to_save = f"{experiment_name}_run{run_idx}_checkpoint_{episode}.pth"
        print(f"\n[SALVATAGGIO OK] Checkpoint saved for Run {run_idx + 1} at episode {episode} in -> {save_dir}")
    elif solved:
        filename_to_save = f"{experiment_name}_run{run_idx}_solved_{episode}.pth"
        print(f"\n[SALVATAGGIO OK] Environment solved for Run {run_idx + 1} at episode {episode} in -> {save_dir}")
    else:
        filename_to_save = f"{experiment_name}_run{run_idx}_failed_{episode}.pth"
        print(f"\n[SALVATAGGIO OK] Training ended for Run {run_idx + 1} without solving at episode {episode} in -> {save_dir}")

    path = os.path.join(save_dir, filename_to_save) #Create the path to the file binding together the save directory and the file name

    #save using a .pth file the parameters of the networks, the optimizer, the configuration's type, the dictionary containing the results for the training's plot, the current episode and run, and the flag for solved (true/false)
    torch.save({
        'local_q': agent.local_qnetwork.state_dict(), 
        'target_q': agent.target_qnetwork.state_dict(),
        'optimizer': agent.optimizer.state_dict(), 
        'config': current_cfg, 
        'results': current_results, 
        "episode": episode, 
        "run_idx": run_idx, 
        "solved": solved}, 
        path)
    
def train_agent(agent, experiment_name, cfg, number_episodes, run_idx,
                env, fixed_states, num_runs, eval_freq = 1):
  #The function takes as input the agent (with its hyperparameters), the experiment name and the configurations, and the max number of episodes for the training and the max number of timesteps t per episode
  
    # Epsilon and parameters are now read dynamically from the 'agent' object properties
    epsilon = agent.epsilon_starting_value #Epsilon has to be initialized before every different training, to assure initial exploration
    epsilon_decay = cfg["epsilon_decay"] #Epsilon decay parameter set for the specific experiments
    rewards = []
    losses = []
    epsilons = []
    episode_times_train = []
    avg_q_list = []
    #These empty lists are used to store the data used for the descriptive plots of the training
    scores_on_100_episodes = deque(maxlen=100) #This is used to assess the scores in the long run; If this value exceeds 200 --> Environment solved
    solved = False #Initialized value as False, it will be used at the end to select the right kind of model save
    

    #Beginning of the single experiment, here we print the name of the current experiment
    print(f"\n--- Starting training for experiment: {experiment_name} (Run {run_idx + 1}/{num_runs}) ---")

    for episode in range(1, number_episodes + 1): 
        state, _ = env.reset() #env.reset() returns the state and additional info; _ is used to store these latter.
        score = 0
        start_episode_time = time.time()
        #Before every episode, the environment is reset to an initial state; also score and the stopwatch are set to 0
        episode_q_values = []

        # Read the maximum timesteps directly from the agent property
        for t in range(agent.maximum_number_timesteps_per_episode): 
            action, max_q = agent.act(state, epsilon) #Here the agent choose an action, based on the current state and epsilon.
                                                       #At the beginning (with high epsilon) there will be more exploration, which will be reduced due to epsilon_decay
            episode_q_values.append(max_q)

            next_state, reward, terminated, truncated, _ = env.step(action) #Now that the action is taken, the environment returns S', R and the value for Done; _ is used to store the additional information (not used)
            done = terminated or truncated #terminated is True if the episode is ended reaching the terminal state (negative or positive)
                                           #truncated if True if the episode ended for other reasons (such as timelimit or if the max_timesteps_per_episode is reached while the lander is still flying in the space)
                                           #At the end of the episode we will obtain values for both (True or False)
                                           # Done will be false if both terminated and truncated are false; otherwise Done = True

            loss = agent.step(state, action, reward, next_state, done) #Compute the loss function and return its value using the step method (which also recall the learn method to update the parameters of the networks)
            if loss is not None:
                losses.append(loss) #This is used to track the loss value over time

            state = next_state #Instantiate the next state
            score += reward #Add the obtained reward to the total score of the episode

            if done: #If done is true (meaning that the max_time_steps is exceeded or the agent landed correctly or not)
                break #Close the episode and go to the next one

        #Record the time taken by the agent to conclude the whole episode
        episode_time = time.time() - start_episode_time #Total time of the episode as difference between the time when the episode start and the one at its end.
        episode_times_train.append(episode_time)

        rewards.append(score)
        scores_on_100_episodes.append(score)
        #Append score to the rewards' list and to the one of score (used to track the possible end of the training)

        #Compute the average Q to track the learning process
        if episode % eval_freq == 0:
            mean_q_this_episode = np.mean(episode_q_values) if episode_q_values else 0.0
            avg_q_list.append(mean_q_this_episode)

        #Epsilon decay update mechanism (Read from agent property)
        epsilon = max(agent.epsilon_ending_value, epsilon_decay * epsilon) #Before every episode epsilon is updated until it reaches the ground floor set by epsilon ending_value
        epsilons.append(epsilon) #Append epsilon to obtain the full record at the end for the plot

        # Update results for the current experiment lists
        current_results = {
            "rewards": rewards,
            "losses": losses,
            "epsilons": epsilons,
            "avg_q": avg_q_list,
            "episode_times": episode_times_train
        }

        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end="")

        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')

        #If statement to check if the agent solved the environment using as measure if during the last 100 episodes, the average score exceeded 200
        if np.mean(scores_on_100_episodes) >= 200.0:
            print(f'\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_on_100_episodes):.2f}')
            solved = True #This is the flag used by the function save_model used to save the experiment as solved
            break #In this case, the training is finished, and the cycle ends here

        # Periodic checkpoint save of the model in the save_directory
        if episode % agent.checkpoint_episode == 0:
            save_model(agent, experiment_name, cfg, current_results, episode, solved, run_idx=run_idx, checkpoint=True)

    # Final save 1: if the agent was able to solve the environment
    if solved:
        save_model(agent, experiment_name, cfg, current_results, episode, solved, run_idx=run_idx, checkpoint=False)
    # Final save 2: if the agent was not able to solve the environment before the max num_episodes
    elif not solved:
        save_model(agent, experiment_name, cfg, current_results, episode, solved, run_idx=run_idx, checkpoint=False)
   

    print(f"--- Training for experiment: {experiment_name} (Run {run_idx + 1}/{num_runs}) ended ---")
    return current_results

# Function used to plot the final grid containing all the graphs computed using the metrics of the training
def plot_experiments_grid_universal(checkpoint_dir, eval_freq=50, output_filename="Complete experiment grid.png"):
    """Group all the runs per experiment and create a 5xN grid with the final results."""
    
    plt.close('all')
    if not os.path.exists(checkpoint_dir): #Check if the given directory really exists, othetwise stop
        print(f" The given directory '{checkpoint_dir}'does not exists.")
        return

    files = os.listdir(checkpoint_dir)
    target_files = [f for f in files if f.endswith(".pth") and "checkpoint" not in f.lower()] #Take all the files not labelled as checkpoint, to maintain only the files related to the end of the training given the running session
    if not target_files: #Look for all the .pth files inside the given directory, otherwise stop
        print(f"No file with .pth was found in '{checkpoint_dir}'.")
        return

    #Group the files by experiment type
    experiments = {}
    for f in target_files:
        name_without_ext = os.path.splitext(f)[0] #remove .pth
        if "_run" in name_without_ext:
            exp_name = name_without_ext.split("_run")[0] #Extract the name of the experiment
        elif "_" in name_without_ext: #Additional control if run "_" is inside the name no extension
            parts = name_without_ext.split("_") # split the string by "_"
            if parts[-1].isdigit() or "run" in parts[-1].lower(): #is a number or run at the end 
                exp_name = "_".join(parts[:-1]) #Maintain everything except the end 
            else:
                exp_name = name_without_ext #No changes are required
        else:
            exp_name = name_without_ext #No changes are required

        if exp_name not in experiments: #If an experiment is not already inside the dictionary
            experiments[exp_name] = [] #Create its list empty and append it inside the dictionary
        experiments[exp_name].append(f)

    exp_list = sorted(list(experiments.keys())) #Sort the list of experiments alphabetically
    num_experiments = len(exp_list) #Compute the length of the experiment list
    print(f"({num_experiments}) Experiments detected: {exp_list}\n")

    metrics_keys = ["rewards", "losses", "avg_q", "epsilons", "episode_times"] #Name of the used metrics
    #Create a dictionary containing all the settings for the plots
    metrics_info = {
        "rewards": {"title": "Mean Reward over the training", "ylabel": "Reward (per ep.)", "color": "tab:blue"},
        "losses": {"title": "Average Loss over the training", "ylabel": "Loss Value", "color": "tab:red"},
        "avg_q": {"title": "Average Q-Value over the training", "ylabel": "Q value", "color": "tab:orange"},
        "epsilons": {"title": "Epsilon Decay over the training", "ylabel": "Epsilon (per ep.)", "color": "tab:green"},
        "episode_times": {"title": "Episode Duration over the training", "ylabel": "Time (s)", "color": "tab:purple"}
    }
    #Subplot used to print all the metrics used to assess the training performances
    fig, axes = plt.subplots(5, num_experiments, figsize=(4.5 * num_experiments, 22))
    if num_experiments == 1: # Create a row vector if only one experiment is present
        axes = np.expand_dims(axes, axis=1)

    for col_idx, exp_name in enumerate(exp_list): 
        run_files = experiments[exp_name] #Take all the different runs of an experiments
        runs_data = [] #Empty list to store the data before computing the averages
        
        #Try to open all the files of an experiment
        for file_name in run_files:
            path = os.path.join(checkpoint_dir, file_name) #Extract the complete path to the file
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False) #load the complete file 
                if 'results' in checkpoint: #if the results data are found in the checkpoint file 
                    runs_data.append({k: list(v) for k, v in checkpoint['results'].items()}) # convert all the data inside the checkpoint file and associate it with  its specific key (metric)
                del checkpoint
            except Exception as e: 
                print(f" An error occurred while loading the file called: {file_name}: {e}")
        
        if not runs_data: # if the list is not empty continue (if not false)
            continue
        n_runs = len(runs_data) #By looking at the number of files, the number of total runs is given

        max_episodes_global = max(len(run_res["rewards"]) for run_res in runs_data if "rewards" in run_res)

        for row_idx, metric_key in enumerate(metrics_keys): 
            ax = axes[row_idx, col_idx] # [metric, experiment]
            info = metrics_info[metric_key] #Retrieve the informations used to plot the graph of that metric
            all_metric_values = [run_res[metric_key] for run_res in runs_data if metric_key in run_res]
            #Take all the values for a metric given a run if it was computed 
            if not all_metric_values: #Continue if the list is not empty
                continue

            max_points = max(len(v) for v in all_metric_values) # Take the max number of episodes to avoid length problems
            padded_values = [] #List which will contains the modified vectors 
            for metric_values in all_metric_values: 
                if len(metric_values) < max_points: # if the list is shorter than the max reached value it will be enlarged by adding zero elements at the end, using the difference between the list and the max at the end with the last registered value
                    padded_values.append(np.pad(metric_values, (0, max_points - len(metric_values)), 'edge'))
                else:
                    padded_values.append(metric_values) # do nothing because max_values == len(metric_values)

            padded_values = np.array(padded_values) #Create an array suitable for the computations of numpy
            mean_metric = np.mean(padded_values, axis=0) #Compute the mean of the metric

            
            if n_runs > 1: #Compute the confidence intervals only if more than 1 runs have been conducted 
                sem_metric = stats.sem(padded_values, axis=0) #Compute the standard error of the mean for every episode
                confidence_interval = sem_metric * stats.t.ppf((1 + 0.95) / 2., n_runs - 1) #Compute the continuous confidence interval at 95% using t-student distribution for all the episodes
                confidence_interval = np.nan_to_num(confidence_interval) #Transform eventual NaN to 0s
            else:
                confidence_interval = np.zeros_like(mean_metric) #Otherwise, the intervals will be not computed (only 1 run)

            #Particular case where a metric is not compute at each episode
            if metric_key == "avg_q" and len(mean_metric) < max_episodes_global:
                x_axis = np.arange(1, len(mean_metric) + 1) * eval_freq
                x_axis = x_axis[:len(mean_metric)]
            else:
                x_axis = np.arange(1, len(mean_metric) + 1) #Otherwise, act as the other metrics

            ax.plot(x_axis, mean_metric, color=info["color"], lw=2, label=f"Average ({n_runs} run)") #Print the mean value across all the episodes
            if n_runs > 1: #If more than one runs have been done, attach the confidence intervals
                ax.fill_between(x_axis, mean_metric - confidence_interval, mean_metric + confidence_interval, 
                                color=info["color"], alpha=0.18)

            ax.grid(True, linestyle='--', alpha=0.5) #Activate the grid
            if row_idx == 0: #Set the name of the experiment
                ax.set_title(f"ESP: {exp_name}", fontsize=11, fontweight='bold', pad=10)
            if col_idx == 0: #Set the metric name 
                ax.set_ylabel(info["ylabel"], fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel(info["title"], fontsize=9, alpha=0.7)

            
            if metric_key == "losses": #The losses have a different x-axis (number of steps instead of training)
                ax.set_xlabel("Number of training steps", fontsize=9)
            else:
                ax.set_xlabel("Number of episodes", fontsize=9) #Otherwise, use the number of episodes (the one x-axis for all the other metrics)
            ax.legend(loc="best", fontsize=8)

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Grid graph successfully saved as: '{output_filename}'") #Save the graph as png
    plt.show() #Plot the final graph

#Function used to compute the final test, used for the comparisons
def test_all_experiments_aggregated(checkpoint_dir, env_name='LunarLander-v3', state_size=8, action_size=4, num_test_episodes=200, success_threshold=200):
    """Executes all the test and print the results inside a Pandas table."""
    #Initial check for the path existence, otherwise block all the test
    if not os.path.exists(checkpoint_dir):
        print(f"'{checkpoint_dir}' Does not exists!, the procedure will be stopped!")
        return

    files = os.listdir(checkpoint_dir)
    target_files = [f for f in files if f.endswith(".pth") and "checkpoint" not in f.lower()] #Take all the non checkpoint files with .pth at the end for this test computation
    if not target_files: #If no final files are found block everywhing, the agent does not solved the environment using the max_number of episodes set at the beginning
        print(f"No eligible file was found, the procedure will be stopped!")
        return

    experiments = {}
    for f in target_files:
        name_without_ext = os.path.splitext(f)[0] #Remove .pth
        if "_run" in name_without_ext:
            exp_name = name_without_ext.split("_run")[0] #remove the "run" and extract the name of the experiment
        elif "_" in name_without_ext: #If "_" are found
            parts = name_without_ext.split("_") #Split the string by "_"
            if parts[-1].isdigit() or "run" in parts[-1].lower(): #If the end is composed by numbers of "run" is present
                exp_name = "_".join(parts[:-1]) #Maintain everything except the end
            else:
                exp_name = name_without_ext
        else:
            exp_name = name_without_ext

        if exp_name not in experiments: # If this experiment is new, create a key,val inside the dictionary
            experiments[exp_name] = []
        experiments[exp_name].append(f)

    exp_list = sorted(list(experiments.keys())) #Create a list of the experiments name and sort it alphabetically
    print(f" The test simulation for {len(exp_list)} experiments is begun...")

    device = torch.device("cpu") #(GPU or CPU)
    env = gym.make(env_name) #Set up the right environment (Lunar-Lender V3)
    rows_output = [] #Empty list for the test results

    for exp_name in exp_list: #Run the test for all the experiments found
        run_files = experiments[exp_name] # Run the test for the different agents
        exp_success_rates, exp_mean_rewards, exp_std_rewards = [], [], [] #Empty lists for the results

        for file_name in run_files: #For every file with solved environment
            path = os.path.join(checkpoint_dir, file_name) #Create the complete path to the file
            try:
                model = Network(state_size, action_size).to(device) #Initialize the agent using the environment parameter
                checkpoint = torch.load(path, map_location=device, weights_only=False) #Load the agent specific for the path (and so the experiment given)
                model.load_state_dict(checkpoint['local_q']) 
                model.eval()#Load the q network and put to evaluation mode
                run_rewards = [] 
                run_successes = 0
                
                for _ in range(num_test_episodes): #Run this cycle for the number of given tests
                    state, _ = env.reset() #Reset the environment everytime at the beginning of the game
                    done = False #The episode is active
                    total_reward = 0 #Beginning reward
                    while not done: #Until done is not truncated or terminated
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device) #Take the state from the environment ad add a dimension, to be suitable for the network's computations
                        with torch.no_grad(): #To block every possible gradients' computations
                            action = model(state_t).argmax().item() #Act greedily because now epsilon == 0
                        next_state, reward, terminated, truncated, _ = env.step(action) #Gather from the environment the informations now that an action is taken by the agent
                        total_reward += reward #Update the reward
                        state = next_state #Update the new state
                        done = terminated or truncated #Check if the run is over
                    
                    run_rewards.append(total_reward) #Append the computed statistics
                    if total_reward >= success_threshold:
                        run_successes += 1 #Update the number of successes
                
                #When the whole cycle is ended, compute the statistics for that run
                exp_success_rates.append(run_successes / num_test_episodes)
                exp_mean_rewards.append(np.mean(run_rewards))
                exp_std_rewards.append(np.std(run_rewards))
                
            except Exception as e:
                print(f"Error testing file {file_name}: {e}")

        #If the file is exists, append the mean values
        if exp_mean_rewards:
            rows_output.append({
                "Experiment": exp_name,
                "Runs Tested": len(exp_mean_rewards),
                "Mean Success Rate": f"{np.mean(exp_success_rates)*100:.1f}%",
                "Global Mean Reward": f"{np.mean(exp_mean_rewards):.2f}",
                "Average Std Reward": f"{np.mean(exp_std_rewards):.2f}"
            })

    env.close()
    df_results = pd.DataFrame(rows_output) #Create a final df with results and print it
    display(df_results)
    return df_results