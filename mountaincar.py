import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3  # Learning Rate
env = gym.make("MountainCar-v0")
env.reset()  # Set the environment to a known reset state
goal_steps = 200
score_requirement = -1*goal_steps  # Score goes down by 1 for every step taken without winning
initial_games = 10000


def some_random_games_first(num_games=5, frames_per_game=200):
    """
    Drives a random agent to play a given number of games, with a given number of frames allotted to each game.
    :return: None
    """
    # Each of these is its own game.
    for episode in range(num_games):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(frames_per_game):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # This executes an action in the environment
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population(initial_games=10000, goal_steps=500, score_requirement=-200, num_possible_actions=3, save=False):
    # [OBS, MOVES]
    training_data = []
    scores = []  # All scores
    accepted_scores = []  # Scores that met our threshold
    for _ in range(initial_games):  # iterate through however many games we want:
        score = 0
        game_memory = []  # Moves specifically from this environment
        prev_observation = []  # previous observation that we saw
        for _ in range(goal_steps):  # Run for "goal_steps" frames
            # choose random action (0: Push left, 1: No push, 2: Push right)
            action = random.randrange(0, num_possible_actions)
            observation, reward, done, info = env.step(action)  # Run the action, retrieve data

            # The observation is returned FROM the action
            # Store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save every move we made
        # NOTE the reinforcement methodology here: all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # Convert to one-hot (this is the output layer for our neural network)
                output = num_to_one_hot(data[1], num_possible_actions)

                training_data.append([data[0], output])  # Save our training data

        env.reset()  # Reset env to play again
        scores.append(score)  # Save overall scores

    if save:
        training_data_save = np.array(training_data)  # Save for future reference
        np.save('mountain_car_saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def num_to_one_hot(number, number_elements=3):
    one_hot = [0 for _ in range(number_elements)]  # Initialize an array of all zeroes.
    if number >= 0:
        one_hot[number] = 1  # Assign the one hot element
    else:
        raise IndexError("%r cannot be converted to a one hot list.")
    return one_hot


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    # incoming = network, n_units=128,256,512...
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)  # incoming = network, keep_prob = 0.8

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


def run_model(model, goal_steps=500, num_actions=3):
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, num_actions)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)

    print('Average Score:', sum(scores) / len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    print(score_requirement)


# Note: Setting goal steps to 200 and the score req caused the cart to not care what it did
# because it ALWAYS reached -200. Goal steps must ALWAYS be greater than the abs val of score req.
if __name__ == "__main__":
    initial_games = 10000
    goal_steps = 500
    score_requirement = -200

    training_data = initial_population(initial_games, goal_steps, score_requirement)
    model = train_model(training_data)
    input("Press enter to continue...")
    run_model(model, goal_steps)
