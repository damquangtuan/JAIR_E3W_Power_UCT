import numpy as np
import matplotlib.pyplot as plt


n_exp = 5
n_trees = 5
n_simulations = 10000
# k_heat = [50, 100, 200]  #[50, 100, 200]
# d_heat = [1, 2] # [50, 100, 200]
k_heat = [2, 4, 6, 8, 10, 12, 14, 16]  #[50, 100, 200]
d_heat = [1, 2, 3, 4] # [50, 100, 200]
# k = [50, 50, 100, 100, 200, 200]
# d = [1, 2, 1, 2, 1, 2]
k = [10, 16, 12, 14, 16]
d = [1, 2, 3, 3, 4]
# k = [50, 100, 200, 50, 100, 200]
# d = [1, 1, 1, 2, 2, 2]
exploration_coeff = .1
tau = .1
# algs = ['uct', 'ments', 'rents', 'tents']

algs = ['alpha-divergence']

algs_legend = [r"$\alpha$=1.0(MENTS)", r"$\alpha$=1.5", r"$\alpha$=2.0(TENTS)", r"$\alpha$=4.0", r"$\alpha$=8.0", r"$\alpha$=16.0"]

alphas = [1, 1.5, 2, 4, 8, 16]

folder_name = 'logs_bk_alpha_divergence/expl_%.2f_tau_%.2f' % (exploration_coeff, tau)

# PLOTS
plt.figure()

count_plot = 0
for kk, dd in zip(k, d):
    max_diff = 0
    max_diff_uct = 0
    max_regret = 0
    for alpha in alphas:
        for alg in algs:
            subfolder_name = folder_name + '/k_%d_d_%d' % (kk, dd)
            diff = np.load(subfolder_name + '/diff_%s_%f.npy' % (alg,alpha))
            avg_diff = diff.mean(0)
            plt.subplot(3, len(k), 1 + count_plot % len(k))
            plt.title('k=%d  d=%d' % (kk, dd), fontsize='xx-large')
            plt.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            plt.yticks(fontsize='xx-large')
            if count_plot == 0:
                plt.ylabel(r'$\varepsilon_\Omega$', fontsize='xx-large')
            plt.plot(avg_diff, linewidth=3)
            err = 2 * np.std(diff.reshape(n_exp * n_trees, n_simulations),
                             axis=0) / np.sqrt(n_exp * n_trees)
            plt.fill_between(np.arange(n_simulations), avg_diff - err, avg_diff + err,
                             alpha=.5)
            max_diff = max(max_diff, avg_diff.max())

            diff_uct = np.load(subfolder_name + '/diff_uct_%s_%f.npy' % (alg,alpha))
            avg_diff_uct = diff_uct.mean(0)
            plt.subplot(3, len(k), len(k) + 1 + count_plot % len(k))
            plt.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            plt.yticks(fontsize='xx-large')
            if count_plot == 0:
                plt.ylabel(r'$\varepsilon_{UCT}$', fontsize='xx-large')
            plt.plot(avg_diff_uct, linewidth=3)
            err = 2 * np.std(diff_uct.reshape(n_exp * n_trees, n_simulations),
                             axis=0) / np.sqrt(n_exp * n_trees)
            plt.fill_between(np.arange(n_simulations), avg_diff_uct - err,
                             avg_diff_uct + err, alpha=.5)
            max_diff_uct = max(max_diff_uct, avg_diff_uct.max())

            regret = np.load(subfolder_name + '/regret_%s_%f.npy' % (alg,alpha))
            avg_regret = regret.mean(0)
            plt.subplot(3, len(k), 2 * len(k) + 1 + count_plot % len(k))
            if count_plot == 0:
                plt.ylabel(r'$R$', fontsize='xx-large')
            plt.plot(avg_regret, linewidth=3)
            err = 2 * np.std(regret.reshape(n_exp * n_trees, n_simulations),
                             axis=0) / np.sqrt(n_exp * n_trees)
            plt.fill_between(np.arange(n_simulations), avg_regret - err,
                             avg_regret + err, alpha=.5)
            max_regret = max(max_regret, avg_regret.max())
            plt.xlabel('# Simulations', fontsize='xx-large')
            plt.xticks([0, 5000, 10000], ['0', '5e3', '10e3'], fontsize='xx-large')
            plt.yticks(fontsize='xx-large')
            plots = [max_diff, max_diff_uct, max_regret]

    for i in range(3):
        plt.subplot(3, len(k), count_plot + 1 + i * len(k))
        plt.grid()
        plt.ylim(0, plots[i])
        
    count_plot += 1

plt.subplot(3, len(k), 3 * len(k) - 2)
plt.legend([alg for alg in algs_legend], fontsize='xx-large', loc="upper center", bbox_to_anchor=(0.3, -0.3), ncol=len(algs_legend), frameon=False)

# HEATMAPS
diff = np.load(folder_name + '/diff_heatmap.npy')
diff_uct = np.load(folder_name + '/diff_uct_heatmap.npy')
regret = np.load(folder_name + '/regret_heatmap.npy')

diffs = [diff, diff_uct, regret]
titles_diff = [r'$\varepsilon_\Omega$', r'$\varepsilon_{UCT}$', 'R']
for t, d in zip(titles_diff, diffs):
    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(t, fontsize='xx-large')
    max_d = d.max()
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(d[i], cmap=plt.get_cmap('inferno'))
        ax.set_title(algs_legend[i], fontsize='xx-large')
        ax.set_xticks(np.arange(len(d_heat)))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-large')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize('xx-large')
        ax.set_yticks(np.arange(len(k_heat)))
        ax.set_xticklabels(d_heat)
        ax.set_yticklabels(k_heat)
        im.set_clim(0, max_d)
    cb_ax = fig.add_axes([.7, .15, .015, .6])
    cbar = fig.colorbar(im, cax=cb_ax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize('xx-large')

plt.show()
