import config
from environment import Environment, Quit
import random
import numpy as np
from environment import Action
import matplotlib.pyplot as pyplot
import pandas
import seaborn
def get_action_eps_greedy_policy(environment, q_tab, st, eps):
    prob = random.uniform(0, 1)
    return np.argmax(q_tab[st]) if prob > eps else environment.get_random_action()

def train(num_episodes, max_steps, lr, gamma, eps_min, eps_max, eps_dec_rate, env):
    avg_returns = []
    avg_steps = []
    vrsta = len(environment.field_map)
    kolona = len(environment.field_map[0])
    q_tab = np.zeros((vrsta*kolona,4))
    for episode in range(num_episodes):
        avg_returns.append(0.)
        avg_steps.append(0)
        eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)
        environment.reset()
        st = environment.get_agent_position()
        for step in range(max_steps):
            staro_stanje = st[0]*kolona+st[1]
            act = get_action_eps_greedy_policy(environment, q_tab, staro_stanje, eps)
            new_st, rew, done = environment.step(Action(act))
            novo_stanje = new_st[0]*kolona+new_st[1]

            akcija = act
            if (act==Action.UP):
                akcija=0
            if (act==Action.LEFT):
                akcija=1
            if (act==Action.DOWN):
                akcija=2
            if (act==Action.RIGHT):
                akcija=3


            q_tab[staro_stanje][akcija] = q_tab[staro_stanje][akcija] + lr * (rew + gamma * np.max(q_tab[novo_stanje]) - q_tab[staro_stanje][akcija])

            if done:
                avg_steps[-1] += step + 1
                avg_returns[-1] += rew
                break
            st[0] = new_st[0]
            st[1] = new_st[1]
    return q_tab, avg_returns, avg_steps

def line_plot(data, name, show,episodes,chunk):
    pyplot.figure(f'Average {name} per episode: {np.mean(data):.2f}')
    print(f'Average {name} per episode: {np.mean(data):.2f}')
    df = pandas.DataFrame({
        name: [np.mean(data[i * chunk:(i + 1) * chunk])
    for i in range(episodes // chunk)],
    'episode': [chunk * i
    for i in range(episodes // chunk)]})
    plot = seaborn.lineplot(data=df, x='episode', y=name, marker='o',
    markersize=5, markerfacecolor='red')
    plot.get_figure().savefig(f'{name}.png')
    if show:
        pyplot.show()



environment = Environment(f'maps/map.txt')
try:

    #Trening agenta
    environment.render(config.FPS)
    Q_tab,avg_reward,avg_steps = train(7000,100,0.05,1,0.005,1.0,0.001,environment)

    #Vizuelizacija rezultata treniranja

    print("\n\n")
    line_plot(avg_reward,"reward",True,7000,100)

    line_plot(avg_steps, "steps", True, 7000, 100)

    #Prikazivanje optimalne putanje i na kraju ispis nagrade ove simulacije
    st = environment.reset()
    st = environment.get_agent_position()
    vrsta = len(environment.field_map)
    kolona = len(environment.field_map[0])
    staro_stanje = st[0]*kolona+st[1]
    step_cnt=0
    ep_rew=0
    environment.render(config.FPS)
    while True:
        act = np.argmax(Q_tab[staro_stanje])
        new_st, rew, done = environment.step(Action(act))
        st[0]=new_st[0]
        st[1]=new_st[1]
        k = environment.get_agent_position()
        #environment.render(config.FPS)
        staro_stanje = st[0] * kolona + st[1]
        step_cnt += 1
        ep_rew += rew
        environment.render(config.FPS)
        if done:
            break

    print(f"Ukupna nagrada je jednaka {ep_rew}")
except Quit:
    pass


