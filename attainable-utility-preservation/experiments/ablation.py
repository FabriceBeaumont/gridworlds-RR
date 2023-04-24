from __future__ import print_function
from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent
from environment_helper import *
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings


def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    if len(framesets) == 7:
        axs = [plt.subplot(3, 3, 2),
               plt.subplot(3, 3, 4), plt.subplot(3, 3, 5), plt.subplot(3, 3, 6),
               plt.subplot(3, 3, 7), plt.subplot(3, 3, 8), plt.subplot(3, 3, 9)]
    else:
        _, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    max_runtime = max([len(frames) for _, frames in framesets])
    ims, zipped = [], zip(framesets, axs if len(framesets) > 1 else [axs])  # handle 1-agent case
    for i in range(max_runtime):
        ims.append([])
        for (agent_name, frames), ax in zipped:
            if i == 0:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(agent_name)
            ims[-1].append(ax.imshow(frames[min(i, len(frames) - 1)], animated=True))
    return animation.ArtistAnimation(plt.gcf(), ims, interval=400, blit=True, repeat_delay=200)


def run_game(game, kwargs):
    render_fig, render_ax = plt.subplots(1, 1)
    render_fig.set_tight_layout(True)
    render_ax.get_xaxis().set_ticks([])
    render_ax.get_yaxis().set_ticks([])
    game.variant_name = game.name + '-' + str(kwargs['level'] if 'level' in kwargs else kwargs['variant'])
    print(game.variant_name)

    start_time = datetime.datetime.now()
    movies = run_agents(game, kwargs, render_ax=render_ax)

    # Save first frame of level for display in paper
    render_ax.imshow(movies[0][1][0])
    render_fig.savefig(os.path.join(os.path.dirname(__file__), 'level_imgs', game.variant_name + '.pdf'),
                       bbox_inches='tight', dpi=350)
    plt.close(render_fig.number)

    print("Training finished; {} elapsed.\n".format(datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join(os.path.dirname(__file__), 'gifs', game.variant_name + '.gif'),
             writer='imagemagick', dpi=350)
    plt.show()


def run_agents(env_class, env_kwargs, render_ax=None):
    """
    Generate and run agent variants.

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    :param render_ax: PyPlot axis on which rendering can take place.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)
    model_free = ModelFreeAUPAgent(env, trials=1)
    state = (ModelFreeAUPAgent(env, state_attainable=True, trials=1))
    movies, agents = [], [ModelFreeAUPAgent(env, num_rewards=0, trials=1),  # vanilla
                          AUPAgent(attainable_Q=model_free.attainable_Q, baseline='start'),
                          AUPAgent(attainable_Q=model_free.attainable_Q, baseline='inaction'),
                          AUPAgent(attainable_Q=model_free.attainable_Q, deviation='decrease'),
                          AUPAgent(attainable_Q=state.attainable_Q, baseline='inaction', deviation='decrease', N=500),  # RR
                          model_free,
                          AUPAgent(attainable_Q=model_free.attainable_Q)  # full AUP
                          ]

    for agent in agents:
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, render_ax=render_ax)
        movies.append((agent.name, frames))
        print(agent.name, perf)

    return movies


games = [(conveyor.ConveyorEnvironment, {'variant': 'vase'}),
         (conveyor.ConveyorEnvironment, {'variant': 'sushi'}),
         (burning.BurningEnvironment, {'level': 0}),
         (burning.BurningEnvironment, {'level': 1}),
         (box.BoxEnvironment, {'level': 0}),
         (sushi.SushiEnvironment, {'level': 0}),
         (vase.VaseEnvironment, {'level': 0}),
         (dog.DogEnvironment, {'level': 0}),
         (survival.SurvivalEnvironment, {'level': 0})
         ]

# Plot setup
plt.style.use('ggplot')

# Get individual game ablations
for (game, kwargs) in games:
    run_game(game, kwargs)
