from __future__ import print_function
from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent
from environment_helper import *
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

settings = [{'label': r'$\gamma$', 'iter': [1 - 2 ** (-n) for n in range(3, 11)],
             'keyword': 'discount'},
            {'label': r'$\lambda$', 'iter': 1/np.arange(.001,3.001,.3), 'keyword': 'lambd'},
            {'label': r'$|\mathcal{R}|$', 'iter': range(0, 50, 5), 'keyword': 'num_rewards'}]
settings[0]['iter_disp'] = ['{0:0.3f}'.format(1 - 2 ** (-n)).lstrip("0") for n in range(3, 11)]
settings[1]['iter_disp'] = ['{0:0.1f}'.format(round(l, 1)).lstrip("0") for l in settings[1]['iter']][::-1]
settings[2]['iter_disp'] = settings[2]['iter']

games = [(box.BoxEnvironment, {'level': 0}),
         (dog.DogEnvironment, {'level': 0}),
         (survival.SurvivalEnvironment, {'level': 0}),
         (conveyor.ConveyorEnvironment, {'variant': 'vase'}),
         (sushi.SushiEnvironment, {'level': 0}),
         ]


def make_charts():
    colors = {'box':      [v / 1000. for v in box.GAME_BG_COLOURS[box.BOX_CHR]],
              'dog':      [v / 1000. for v in dog.GAME_BG_COLOURS[dog.DOG_CHR]],
              'survival': [v / 1000. for v in survival.GAME_BG_COLOURS[survival.BUTTON_CHR]],
              'conveyor': [v / 1000. for v in conveyor.GAME_BG_COLOURS[conveyor.OBJECT_CHR]],
              'sushi':    [v / 1000. for v in sushi.GAME_BG_COLOURS[sushi.SUSHI_CHR]]}

    order = ['box', 'dog', 'survival', 'conveyor', 'sushi']
    new_names = ['options', 'damage', 'correction', 'offset', 'interference']

    plt.style.use('ggplot')
    fig = plt.figure(1)
    axs = [fig.add_subplot(3, 1, plot_ind + 1) for plot_ind in range(3)]
    fig.set_size_inches(7, 4, forward=True)
    for plot_ind, setting in enumerate(settings):
        counts = np.load(os.path.join(os.path.dirname(__file__), 'plots', 'counts-' + setting['keyword'] + '.npy'),
                         encoding="latin1")[()]

        stride = 3 if setting['keyword'] == 'num_rewards' else 2
        ax = axs[plot_ind]
        ax.tick_params(axis='x', which='minor', bottom=False)

        ax.set_xlabel(setting['label'])
        if setting['keyword'] == 'lambd':
            ax.set_ylabel('Trials')
            for key in counts.keys():
                counts[key] = counts[key][::-1]
        x = np.array(range(len(setting['iter'])))

        tick_pos, tick_labels = [], []
        text_ind, text = [], []

        width = .85
        offset = (len(setting['iter']) + 1)

        ordered_counts = [(name, counts[name]) for name in order]
        for x_ind, (game_name, data) in enumerate(ordered_counts):
            tick_pos.extend(list(x + offset * x_ind))
            text_ind.append((len(setting['iter']) -.75) / 2 + offset * x_ind)

            tick_labels.extend([setting['iter_disp'][i] if i % stride == 0 else '' for i in range(len(setting['iter']))])
            if setting['keyword'] == 'discount':
                text.append(r'$\mathtt{' + new_names[x_ind].capitalize() + '}$')

            for ind, (label, color) in enumerate([("Side effect,\nincomplete", (.3, 0, 0)),
                                                  ("Side effect,\ncomplete", (.65, 0, 0)),
                                                  ("No side effect,\nincomplete", "xkcd:gray"),
                                                  ("No side effect,\ncomplete", (0.0, .624, 0.42))]):
                ax.bar(x + offset * x_ind, data[:, ind], width, label=label, color=color,
                       bottom=np.sum(data[:, :ind], axis=1) if ind > 0 else 0, zorder=3)

        # Wrangle ticks and level labels
        ax.set_xlim([-1, tick_pos[-1] + 1])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xticks(text_ind, minor=True)
        ax.set_xticklabels(text, minor=True, fontdict={"fontsize": 8})
        for lab in ax.xaxis.get_minorticklabels():
            lab.set_y(1.34)
        ax.tick_params(axis='both', width=.5, labelsize=7)

        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:4][::-1], labels[:4][::-1], fontsize='x-small', loc='upper center', facecolor='white',
               edgecolor='white', ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .97), h_pad=0.15)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'all.pdf'), bbox_inches='tight')

    # Plot of episodic performance data
    perf = np.load(os.path.join(os.path.dirname(__file__), 'plots', 'performance.npy'), encoding="latin1")[()]

    eps_fig, eps_ax = plt.subplots()
    eps_fig.set_size_inches(7, 2, forward=True)
    eps_ax.set_xlabel('Episode')
    eps_ax.set_ylabel('Performance')
    eps_ax.set_xlim([-150, 6150])
    eps_ax.set_yticks([-1, 0, 1])

    for ind, name in enumerate(order):
        eps_ax.plot(range(0, len(perf[name][0]) * 10, 10),
                    np.average(perf[name], axis=0), label=r'$\mathtt{' + new_names[ind].capitalize() + '}$',
                    color=colors[name], zorder=3)

    # Mark change in exploration strategy
    eps_ax.axvline(x=4000, color=(.4, .4, .4), zorder=1, linewidth=2, linestyle='--')
    eps_ax.legend(loc='upper center', facecolor='white', edgecolor='white', ncol=len(order),
                  bbox_to_anchor=(0.5, 1.2))

    eps_fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'episodes.pdf'), bbox_inches='tight')

    plt.show()


def run_exp(ind):
    setting = settings[ind]
    print(setting['label'])

    counts, perf = dict(), dict()
    for (game, kwargs) in games:
        counts[game.name] = np.zeros((len(setting['iter']), 4))
        for (idx, item) in enumerate(setting['iter']):
            env = game(**kwargs)
            model_free = ModelFreeAUPAgent(env, trials=50, **{setting['keyword']: item})
            if setting['keyword'] == 'lambd' and item == ModelFreeAUPAgent.default['lambd']:
                perf[game.name] = model_free.performance
            counts[game.name][idx, :] = model_free.counts[:]
            print(game.name.capitalize(), setting['keyword'], item, model_free.counts)
    np.save(os.path.join(os.path.dirname(__file__), 'plots', 'performance'), perf)
    np.save(os.path.join(os.path.dirname(__file__), 'plots', 'counts-' + setting['keyword']), counts)


if __name__ == '__main__':
    p = Pool(3)
    p.map(run_exp, range(len(settings)))
    make_charts()
