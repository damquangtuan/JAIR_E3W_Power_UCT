import numpy as np
from scipy.special import logsumexp

from scipy.stats import norm


class MCTS:
    def __init__(self, exploration_coeff, algorithm, tau, alpha, step_size, gamma, update_type):
        self._exploration_coeff = exploration_coeff
        self._algorithm = algorithm
        self._tau = tau
        self._alpha = alpha
        self._step_size = step_size
        self._gamma = gamma # discount factor
        self._update_type = update_type
        if algorithm == 'alpha-divergence' and alpha == 1:
            self._algorithm = 'ments'
        if algorithm == 'alpha-divergence' and alpha == 2:
            self._algorithm = 'tents'

    def run(self, tree_env, n_simulations):
        v_hat = np.zeros(n_simulations)
        regret = np.zeros_like(v_hat)
        for i in range(n_simulations):
            tree_env.reset()
            v_hat[i], regret[i] = self._simulation(tree_env)

        return v_hat, regret.cumsum()

    @staticmethod
    def _compute_prob_max(mean_list, sigma_list):
        n_actions = len(mean_list)
        lower_limit = mean_list - 8 * sigma_list
        upper_limit = mean_list + 8 * sigma_list
        epsilon = 1e-5
        _epsilon = 1e-25
        n_trapz = 100
        x = np.zeros(shape=(n_trapz, n_actions))
        y = np.zeros(shape=(n_trapz, n_actions))
        integrals = np.zeros(n_actions)
        for j in range(n_actions):
            if sigma_list[j] < epsilon:
                p = 1
                for k in range(n_actions):
                    if k != j:
                        p *= norm.cdf(mean_list[j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = p
            else:
                x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
                y[:, j] = norm.pdf(x[:, j],loc=mean_list[j], scale=sigma_list[j] + _epsilon)
                for k in range(n_actions):
                    if k != j:
                        y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * (y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))

        with np.errstate(divide='raise'):
            try:
                return integrals / (np.sum(integrals))
            except FloatingPointError:
                print(integrals)
                print(mean_list)
                print(sigma_list)
                input()

    def _simulation(self, tree_env):
        path = self._navigate(tree_env)
        leaf_node = tree_env.tree.nodes[path[-1][1]]
        reward = tree_env.rollout(path[-1][1])

        leaf_node['V'] = (leaf_node['V'] * leaf_node['N'] + reward) / (leaf_node['N'] + 1)
        leaf_node['N'] += 1
        for e in reversed(path):
            current_node = tree_env.tree.nodes[e[0]]
            next_node = tree_env.tree.nodes[e[1]]
            tree_env.tree[e[0]][e[1]]['Q'] = next_node['V']
            tree_env.tree[e[0]][e[1]]['N'] += 1
            if self._algorithm == 'uct':
                current_node['V'] = (current_node['V'] * current_node['N'] +
                                     tree_env.tree[e[0]][e[1]]['Q']) / (current_node['N'] + 1)
            elif self._algorithm == 'power-uct':
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                counts = np.array(
                    [tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
                qvalues = np.array(
                    [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
                total = np.sum(counts * np.power(qvalues,self._alpha))
                visits = np.sum(counts)
                current_node['V'] = np.power(total/visits,1.0/self._alpha)
            else:
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                qs = np.array(
                    [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
                if self._algorithm == 'ments':
                    current_node['V'] = self._tau * logsumexp(qs / self._tau)
                elif self._algorithm == 'rents':
                    qs_tau = qs / self._tau
                    weighted_logsumexp_qs = qs_tau.max() + np.log(
                        np.sum(current_node['prior'] * np.exp(qs_tau - qs_tau.max()))
                    )
                    current_node['V'] = self._tau * weighted_logsumexp_qs
                elif self._algorithm == 'tents':
                    q_tau = qs / self._tau
                    temp_q_tau = q_tau.copy()
                    sorted_q = np.flip(np.sort(temp_q_tau))
                    kappa = list()
                    for i in range(1, len(sorted_q) + 1):
                        if 1 + i * sorted_q[i-1] > sorted_q[:i].sum():
                            idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                            temp_q_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)
                    sparse_max = q_tau[kappa] ** 2 / 2 - (q_tau[kappa].sum() - 1) ** 2 / (2 * len(kappa) ** 2)
                    sparse_max = sparse_max.sum() + .5
                    current_node['V'] = self._tau * sparse_max
                elif self._algorithm == 'alpha-divergence':
                    q_tau = qs / self._tau
                    temp_q_tau = q_tau.copy()

                    sorted_q = np.flip(np.sort(temp_q_tau))
                    kappa = list()
                    for i in range(1, len(sorted_q) + 1):
                        if 1 + i * sorted_q[i-1] > sorted_q[:i].sum() + i * (1 - (1/(self._alpha-1))):
                            idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                            temp_q_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)
                    c_s_tau = ((q_tau[kappa].sum() - 1) / len(kappa)) + (1 - (1/(self._alpha-1)))
                    max_omega = np.maximum(q_tau - c_s_tau, np.zeros(len(q_tau)))
                    max_omega = np.power(max_omega * (self._alpha - 1), 1/(self._alpha - 1))
                    max_omega = max_omega/np.sum(max_omega)
                    # sparse_max_tmp = max_omega * (q_tau + (1/(self._alpha - 1)) * (1 - max_omega_tmp))
                    sparse_max_tmp = max_omega * q_tau
                    sparse_max = sparse_max_tmp.sum()
                    current_node['V'] = self._tau * sparse_max
                else:
                    raise ValueError

            current_node['N'] += 1

        v_hat = tree_env.tree.nodes[0]['V']
        max_a = self._select(tree_env=tree_env, state=0)
        regret = tree_env.q_root.max() - tree_env.q_root[max_a]
        return v_hat, regret

    def _navigate(self, tree_env):
        state = tree_env.state
        action = self._select(tree_env, state)
        next_state = tree_env.step(action)
        if next_state not in tree_env.leaves:
            return [[state, next_state]] + self._navigate(tree_env)
        else:
            return [[state, next_state]]

    def _select(self, tree_env, state):
        out_edges = [e for e in tree_env.tree.edges(state)]
        n_state_action = np.array(
            [tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
        qs = np.array(
            [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])

        if self._algorithm == 'uct':
            n_state = np.sum(n_state_action)
            if n_state > 0:
                ucb_values = qs + self._exploration_coeff * np.sqrt(
                    np.log(n_state) / (n_state_action + 1e-10)
                )
            else:
                ucb_values = np.ones(len(n_state_action)) * np.inf

            chosen_action = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
            probs = np.zeros_like(ucb_values)
            probs[chosen_action] += 1

            return chosen_action
        elif self._algorithm == 'power-uct':
            n_state = np.sum(n_state_action)
            if n_state > 0:
                ucb_values = qs + self._exploration_coeff * np.sqrt(
                    np.log(n_state) / (n_state_action + 1e-10)
                )
            else:
                ucb_values = np.ones(len(n_state_action)) * np.inf

            chosen_action = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
            probs = np.zeros_like(ucb_values)
            probs[chosen_action] += 1

            return chosen_action
        else:
            n_actions = len(out_edges)
            lambda_coeff = np.clip(self._exploration_coeff * n_actions / np.log(
                np.sum(n_state_action) + 1 + 1e-10), 0, 1)

            if self._algorithm == 'ments':
                q_exp_tau = np.exp(qs / self._tau)
                probs = (1 - lambda_coeff) * q_exp_tau / q_exp_tau.sum() + lambda_coeff / n_actions
            elif self._algorithm == 'rents':
                qs_tau = qs / self._tau
                prior_q_exp_tau = tree_env.tree.nodes[state]['prior'] * np.exp(qs_tau - qs_tau.max())
                probs = (1 - lambda_coeff) * prior_q_exp_tau / (prior_q_exp_tau.sum()) + lambda_coeff / n_actions
            elif self._algorithm == 'tents':
                q_tau = qs / self._tau
                temp_q_tau = q_tau.copy()

                sorted_q = np.flip(np.sort(temp_q_tau))
                kappa = list()
                for i in range(1, len(sorted_q) + 1):
                    if 1 + i * sorted_q[i-1] > sorted_q[:i].sum():
                        idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                        temp_q_tau[idx] = np.nan
                        kappa.append(idx)
                kappa = np.array(kappa)

                max_omega = np.maximum(q_tau - (q_tau[kappa].sum() - 1) / len(kappa),
                                       np.zeros(len(q_tau)))
                probs = (1 - lambda_coeff) * max_omega + lambda_coeff / n_actions
            elif self._algorithm == 'alpha-divergence':
                q_tau = qs / self._tau
                temp_q_tau = q_tau.copy()

                sorted_q = np.flip(np.sort(temp_q_tau))
                kappa = list()
                for i in range(1, len(sorted_q) + 1):
                    if 1 + i * sorted_q[i-1] > sorted_q[:i].sum() + i * (1 - (1/(self._alpha-1))):
                        idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                        temp_q_tau[idx] = np.nan
                        kappa.append(idx)
                kappa = np.array(kappa)
                c_s_tau = ((q_tau[kappa].sum() - 1) / len(kappa)) + (1 - (1/(self._alpha-1)))

                max_omega = np.maximum(q_tau - c_s_tau, np.zeros(len(q_tau)))
                max_omega = np.power(max_omega * (self._alpha - 1), 1/(self._alpha - 1))
                max_omega = max_omega/np.sum(max_omega)
                probs = (1 - lambda_coeff) * max_omega + lambda_coeff / n_actions
            else:
                raise ValueError

            return np.random.choice(np.arange(n_actions), p=probs)
