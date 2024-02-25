##########################
# YOU CAN EDIT THIS FILE #
##########################
# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Imports from this project
import constants
import configuration
from graphics import PathToDraw


class MBNet(torch.nn.Module):
    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(MBNet, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 10 units.
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=16, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=16, out_features=16, dtype=torch.float32)
        # self.layer_3 = torch.nn.Linear(in_features=16, out_features=16, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=16, out_features=2, dtype=torch.float32)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        # layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_2_output)
        return output


def reset_weights(m):
    '''
    This function takes a PyTorch model layer as input
    and reinitializes its weights.
    '''
    # Check if the layer is a Convolutional layer or a Linear layer
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class BCNet(torch.nn.Module):
    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(BCNet, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 10 units.
        self.network = nn.Sequential(
            nn.Linear(2, 64),  # Input layer with input dimension 2
            nn.ReLU(),
            nn.Linear(64, 128),  # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, 2)  # Output layer with output dimension 2
        )

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        return self.network(input)


def point_line_segment_distance(p, p1, p2):
    """Calculate the distance from point p to the line segment defined by points p1 and p2."""
    line_vec = p2 - p1
    p_vec = p - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    p_vec_scaled = p_vec / line_len
    t = np.dot(line_unitvec, p_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = np.linalg.norm(p_vec - nearest)
    return dist


def find_distant_points(array1, array2, threshold):
    diffs = array1[:, np.newaxis, :] - array2  # Shape becomes (n, m, 2)
    squared_distances = np.sum(diffs ** 2, axis=2)  # Sum over the last dimension to get squared distances

    # Check if all distances for a point in array1 are greater than the threshold squared
    is_far_enough = np.all(squared_distances > threshold ** 2, axis=1)

    # Get indices where the condition is True
    indices = np.where(is_far_enough)[0]

    return indices


def shortest_distance_to_path(point, waypoints):
    """Find the shortest distance from a point to a series of waypoints."""
    min_dist = float('inf')
    for i in range(len(waypoints) - 1):
        dist = point_line_segment_distance(point, waypoints[i], waypoints[i + 1])
        if dist < min_dist:
            min_dist = dist
    return min_dist


class Robot:

    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.paths_to_draw = []
        self.dynamics_model_network = MBNet()
        self.inputs_mb = np.array([0])
        self.labels_mb = np.array([0])

        self.inputs_bc = np.array([0])
        self.labels_bc = np.array([0])
        self.behavior_clone_network = BCNet()
        self.demo_count = 0
        # self.demonstration_actions = np.array([])
        self.demonstration_states = np.array([])
        self.curr_path = []
        self.curr_action = []
        self.noise = 0.0

        self.demonstration = True
        self.step = False
        self.reset = False

        self.planned_path = []
        self.planned_actions = []
        self.planning_actions = []
        self.planning_paths = []
        self.planning_path_rewards = []
        self.planning_mean_actions = []

        self.episode = 0

        self.plan_index = 0

    def reset_param(self):
        self.planned_actions = []
        self.planned_path = []
        self.planning_paths = []
        self.planning_path_rewards = []
        self.planning_mean_actions = []
        self.plan_index = 0
        self.curr_path = []
        self.curr_action = []

    def cross_entropy_method_planning(self, robot_current_state, goal_state, path_length=50, num_path=30, iteration=10,
                                      num_elites=5):
        # planning_actions is the full set of actions that are sampled
        self.planning_actions = np.zeros(
            [iteration, num_path, path_length, 2],
            dtype=np.float32)
        # planning_paths is the full set of paths (one path is a sequence of states) that are evaluated
        self.planning_paths = np.zeros(
            [iteration, num_path, path_length, 2],
            dtype=np.float32)
        # planning_path_rewards is the full set of path rewards that are calculated
        self.planning_path_rewards = np.zeros([iteration, num_path])
        # planning_mean_actions is the full set of mean action sequences that are calculated at the end of each
        # iteration (one sequence per iteration)
        self.planning_mean_actions = np.zeros([iteration, path_length, 2],
                                              dtype=np.float32)
        # Loop over the iterations
        # action_mean = self.demonstration_actions
        action_mean = np.zeros((path_length, 2))
        factor = 1

        action_std_dev = np.ones((path_length, 2)) * factor

        # Loop over the iterations
        for iteration_num in range(iteration):
            # In each iteration, a new set of paths will be sampled
            for path_num in range(num_path):
                # For each sampled path, compute a sequence of states by sampling actions from the current action
                # distribution To begin, set the state for the first step in the path to be the robot's actual
                # current state
                current_state = np.copy(robot_current_state)
                for step_num in range(path_length):
                    action = np.random.normal(action_mean[step_num], action_std_dev[step_num])
                    self.planning_actions[iteration_num, path_num, step_num] = action
                    next_state = self.dynamics_model(current_state, action)
                    next_state = next_state.reshape((2,))
                    self.planning_paths[iteration_num, path_num, step_num] = next_state
                    current_state = next_state

                # Calculate the reward for this path
                path_reward = -np.linalg.norm(current_state - goal_state)
                self.planning_path_rewards[iteration_num, path_num] = path_reward

            top_paths_indices = np.argsort(self.planning_path_rewards[iteration_num])[-num_elites:]
            top_actions = self.planning_actions[iteration_num, top_paths_indices]
            action_mean = np.mean(top_actions, axis=0)
            action_std_dev = np.std(top_actions, axis=0) * 0.9

            self.planning_mean_actions[iteration_num] = action_mean

        # Calculate the index of the best path
        index_best_path = np.argmax(self.planning_path_rewards[-1])
        # Set the planned path (i.e. the best path) to be the path whose index is index_best_path
        self.planned_path = self.planning_paths[-1, index_best_path]
        # Set the planned actions (i.e. the best action sequence) to be the action sequence whose index is
        # index_best_path
        self.planned_actions = self.planning_actions[-1, index_best_path]

    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'

        if self.demonstration == True:
            self.demonstration = False
            self.step = True
            return 'demo'

        '''
        if self.demonstration == True and self.demo_count == 2:
            self.demonstration = False
            self.step = True
            self.demo_count += 1
            return 'demo'

        elif self.demonstration == True:
            self.demo_count += 1
            return 'demo'
        '''

        if self.reset == True:
            self.reset = False
            self.step = True
            return 'reset'

        if self.step == True:
            # print(self.planned_actions)
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned

        '''
        if money_remaining < 70 and self.episode == 0:
            self.cross_entropy_method_planning(state, path_length=200)
            print(self.planned_actions)
            action = self.planned_actions[self.plan_index]
            self.plan_index += 1

        elif money_remaining < 70 and self.episode > 0:
            action = self.planned_actions[self.plan_index]
            if self.plan_index == len(self.planned_actions) - 1:
                self.reset = True
                self.reset_param()
                self.episode = 0
                self.train_dynamic()
                return action
            self.plan_index += 1

        else:
            state_tensor = torch.tensor(state.reshape((1, 2)), dtype=torch.float32)
            action = self.behavior_clone_network(state_tensor)
            action = action.detach().cpu().numpy()
            action = action.reshape((2,))
        '''

        # nearest_index, nearest_point = self.find_closest_waypoint(state, self.demonstration_states)



        if shortest_distance_to_path(state, self.demonstration_states) > 3:
            action = (state - self.goal_state)

        else:
            state_tensor = torch.tensor(state.reshape((1, 2)), dtype=torch.float32)
            action = self.behavior_clone_network(state_tensor)
            action = action.detach().cpu().numpy()
            action = action.reshape((2,))
            gaussian = np.random.normal(loc=0, scale=self.noise, size=1)
            action = action + gaussian



        self.episode += 1
        return action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        '''
        if shortest_distance_to_path(state, self.demonstration_states) > 5:
            action = (state - self.goal_state)

        else:
            state_tensor = torch.tensor(state.reshape((1, 2)), dtype=torch.float32)
            action = self.behavior_clone_network(state_tensor)
            action = action.detach().cpu().numpy()
            action = action.reshape((2,))
        '''
        if shortest_distance_to_path(state, self.demonstration_states) > 3:
            action = (state - self.goal_state)

        state_tensor = torch.tensor(state.reshape((1, 2)), dtype=torch.float32)
        action = self.behavior_clone_network(state_tensor)
        action = action.detach().cpu().numpy()
        action = action.reshape((2,))
        return action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens
        if len(self.inputs_mb.shape) == 1:
            self.inputs_mb = np.concatenate((state.reshape((1, 2)), action.reshape((1, 2))), axis=1).reshape((1, 4))
            self.labels_mb = next_state.reshape((1, 2))
        else:
            new_inputs = np.concatenate((state.reshape((1, 2)), action.reshape((1, 2))), axis=1).reshape((1, 4))
            self.inputs_mb = np.concatenate((self.inputs_mb, new_inputs), axis=0)
            self.labels_mb = np.concatenate((self.labels_mb, next_state.reshape((1, 2))), axis=0)

        '''
        if len(self.inputs_bc.shape) == 1:
            self.inputs_bc = state.reshape((1, 2))
            self.labels_bc = action.reshape((1, 2))
        else:
            self.inputs_bc = np.concatenate((self.inputs_bc, state.reshape((1, 2))), axis=0)
            self.labels_bc = np.concatenate((self.labels_bc, action.reshape((1, 2))), axis=0)
        '''
        self.curr_path.append(state)
        self.curr_action.append(action)

        if self.episode == 200 and money_remaining >= 5:
            # self.train_dynamic()
            '''
            self.inputs_bc = self.inputs_bc[0:self.inputs_bc.shape[0] - 200]
            self.labels_bc = self.labels_bc[0:self.labels_bc.shape[0] - 200]
            '''
            self.noise += 2
            self.reset = True
            self.train_imitation()
            self.reset_param()
            self.episode = 0

        if np.linalg.norm(next_state - self.goal_state) < 5 and money_remaining >= 5:
            # self.train_dynamic()
            self.curr_path = np.array(self.curr_path)
            self.curr_action = np.array(self.curr_action)

            if self.episode < self.demonstration_states.shape[0] or self.inputs_bc.shape[0] == 200:
                self.demonstration_states = self.curr_path

            to_append_indice = find_distant_points(self.curr_path, self.inputs_bc, 15)
            to_append = self.curr_path[to_append_indice]
            to_append_label = self.curr_action[to_append_indice]
            self.inputs_bc = np.concatenate((self.inputs_bc, to_append), axis=0)
            self.labels_bc = np.concatenate((self.labels_bc, to_append_label), axis=0)
            # self.behavior_clone_network.apply(reset_weights)
            self.train_imitation()
            print(to_append.shape)
            print(self.inputs_bc.shape)
            self.reset = True
            self.noise = 0.0
            self.reset_param()
            self.episode = 0

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        self.demonstration_actions = demonstration_actions
        self.demonstration_states = demonstration_states
        # TODO: This allows you to process or store a demonstration that the robot has received
        # demonstration_states (199, 2)
        if len(self.inputs_mb.shape) == 1:
            self.inputs_mb = np.concatenate((demonstration_states[0:198], demonstration_actions[0:198]), axis=1)
            self.labels_mb = demonstration_states[1:199]
        else:
            new_inputs = np.concatenate((demonstration_states[0:198], demonstration_actions[0:198]), axis=1)
            self.inputs_mb = np.concatenate((self.inputs_mb, new_inputs), axis=0)
            self.labels_mb = np.concatenate((self.labels_mb, demonstration_states[1:199]), axis=0)

        self.train_dynamic()

        if len(self.inputs_bc.shape) == 1:
            self.inputs_bc = demonstration_states
            self.labels_bc = demonstration_actions[0:199]
        else:
            self.inputs_bc = np.concatenate((self.inputs_bc, demonstration_states), axis=0)
            self.labels_bc = np.concatenate((self.labels_bc, demonstration_states[0:199]), axis=0)

        self.train_imitation()

    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model,
        # although the actual environment dynamics is much more complex

        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        # Make a prediction with the neural network
        network_input = torch.cat((state_tensor, action_tensor), dim=0)
        network_input = torch.unsqueeze(network_input, 0)
        predicted_next_state = self.dynamics_model_network.forward(network_input)[0]
        predicted_next_state = predicted_next_state.detach().cpu().numpy()
        return predicted_next_state

    def train_dynamic(self):
        network = self.dynamics_model_network
        optimiser = torch.optim.Adam(network.parameters(), lr=0.001)
        losses = []
        iterations = []
        input_data = self.inputs_mb.astype(np.float32)
        label_data = self.labels_mb.astype(np.float32)
        for training_iteration in range(1000):
            optimiser.zero_grad()
            # Sample a mini-batch of size 5 from the training data
            minibatch_indices = np.random.choice(range(input_data.shape[0]), input_data.shape[0])
            minibatch_inputs = input_data[minibatch_indices]
            minibatch_labels = label_data[minibatch_indices]
            # Convert the NumPy array into a Torch tensor
            minibatch_input_tensor = torch.tensor(minibatch_inputs)
            minibatch_labels_tensor = torch.tensor(minibatch_labels)
            # Do a forward pass of the network using the inputs batch
            network_prediction = network.forward(minibatch_input_tensor)

            # Compute the loss based on the label's batch
            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)

            # Compute the gradients based on this loss,
            # i.e. the gradients of the loss with respect to the network parameters.
            loss.backward()
            # Take one gradient step to update the network
            optimiser.step()
            # Get the loss as a scalar value
            loss_value = loss.item()
            # Print out this loss

            print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))

            # Store this loss in the list
            losses.append(loss_value)
            # Update the list of iterations
            iterations.append(training_iteration)
            # Plot and save the loss vs iterations graph
        self.dynamics_model_network = network
        # torch.save(self.dynamics_model_network.state_dict(), 'model_initial.pt')

    def train_imitation(self):
        network = self.behavior_clone_network
        optimiser = torch.optim.Adam(network.parameters(), lr=0.01)
        losses = []
        iterations = []
        input_data = self.inputs_bc.astype(np.float32)
        label_data = self.labels_bc.astype(np.float32)
        for training_iteration in range(1000):
            optimiser.zero_grad()
            # Sample a mini-batch of size 5 from the training data
            minibatch_indices = np.random.choice(range(input_data.shape[0] - 1), int(input_data.shape[0]))
            minibatch_inputs = input_data[minibatch_indices]
            minibatch_labels = label_data[minibatch_indices]
            # Convert the NumPy array into a Torch tensor
            minibatch_input_tensor = torch.tensor(minibatch_inputs)
            minibatch_labels_tensor = torch.tensor(minibatch_labels)
            # Do a forward pass of the network using the inputs batch
            network_prediction = network.forward(minibatch_input_tensor)

            # Compute the loss based on the label's batch
            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)

            # Compute the gradients based on this loss,
            # i.e. the gradients of the loss with respect to the network parameters.
            loss.backward()
            # Take one gradient step to update the network
            optimiser.step()
            # Get the loss as a scalar value
            loss_value = loss.item()
            # Print out this loss

            print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
            # Store this loss in the list
            losses.append(loss_value)
            # Update the list of iterations
            iterations.append(training_iteration)
            # Plot and save the loss vs iterations graph
        self.behavior_clone_network = network
        # torch.save(self.behavior_clone_network.state_dict(), 'model_initial.pt')
