import matplotlib.pyplot as plt
import pickle
import numpy as np

# #def lineplotCI(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title, show=False):
def lineplotCI(cutoff, data, 
        #x_data, y_data, sorted_x, low_CI, upper_CI,
        x_label, y_label, title, show=False):
    # Create the plot object

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    for name, color, y_data, low_CI, upper_CI in data:
        plt.plot(range(len(y_data))[cutoff:], y_data[cutoff:],
                 lw = 1, color = color, alpha = 1, label=name)
        # Shade the confidence interval
        plt.fill_between(range(len(y_data))[cutoff:], low_CI[cutoff:], upper_CI[cutoff:],
                         color = color, alpha = 0.4)
        # Label the axes and provide a title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display legend
    plt.legend(loc = 'best')
    if show:
        plt.show()


def get_CI_data(file_names, flip_neg = False):
    data_files  = list(map(lambda x: 'batch/'+x, file_names))

    data =[]
    run0 = pickle.load(open(data_files[0], 'rb'))
    for e in run0:
        data.append([e])

    for f in data_files[1:]:
        run = pickle.load(open(f, 'rb'))
        for i,e in enumerate(run):
            data[i].append(e)
        
    if flip_neg:
        data = -np.array(data)

    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    Z  = 1.960
    lower = mean - Z * std / np.sqrt(len(data_files))
    upper = mean + Z * std / np.sqrt(len(data_files))

    return mean, lower, upper

# GET THE DATA FOR ERROR
lin_ucb_error_data_files = ['errorLinear UCB440799',
                     'errorLinear UCB624516',
                     'errorLinear UCB708433',
                     'errorLinear UCB777131',
                     'errorLinear UCB972487',
                     'errorLinear UCB572277',
                     'errorLinear UCB674927',
                     'errorLinear UCB734551',
                     'errorLinear UCB783172',
                     'errorLinear UCB973196']

lin_ucb_error_data = get_CI_data(lin_ucb_error_data_files, True)

lineplotCI(100, [('Lin UCB',  '#539caf', *lin_ucb_error_data)], 'timestep t', 'cumulative error', 'Error Rate', True)


lin_ucb_regret_files = map(lambda x: x.replace('error', 'regret'), lin_ucb_error_data_files)
lin_ucb_regret_data = get_CI_data(lin_ucb_regret_files)
lineplotCI(100,
        [('Lin UCB',  '#539caf', *lin_ucb_regret_data),
         ('.39x',  '#000000', list(map(lambda x: 0.39 * x, range(0, 5528))),
                              list(map(lambda x: 0.39 * x, range(0, 5528))),
                              list(map(lambda x: 0.39 * x, range(0, 5528)))),
        # ('Always Wrong',  '#000000', range(0, 5528), range(0,5528), range(0,5528))
        ],
        'timestep t', 'cumulative regret', 'Regret', True)
