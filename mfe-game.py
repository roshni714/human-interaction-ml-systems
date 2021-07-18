import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy import integrate


def phi(x, beta):
    ones = torch.ones(x.shape) # n x 1
    new_x = torch.hstack((x, ones)) #n x 2
    result = torch.sum(new_x *beta, axis=1) #n x 1
    return torch.clamp(result.reshape(-1, 1), min=0.) #n x 1

def allocation(x, beta, dist_mean, budget):
    denom = dist_mean
    res = (phi(x, beta)/(denom+ 1e-5)) * budget
    return res

def get_dist_mean(x, beta):
    phis  = phi(x, beta)
    return torch.sum(phis)

def policy_loss(w, theta):
    return torch.pow(w - theta, 2)

def gaming_cost(theta, theta_prime, game_weight=0.5):
    return torch.pow(theta-theta_prime, 2) * game_weight

def get_agent_features(theta, theta_tilde, beta, dist_mean, budget):
    no_gaming_cost = theta - allocation(theta, beta, dist_mean, budget) + gaming_cost(theta, theta)
    g_cost = theta - allocation(theta_tilde, beta, dist_mean, budget) + gaming_cost(theta, theta_tilde)
    cost_table = torch.hstack((no_gaming_cost, g_cost))
    idx = torch.argmin(cost_table, axis=1)
    new_features = theta.clone()
    new_features[torch.where(idx == 1)] = theta_tilde.clone()[torch.where(idx==1)]
    prop_gaming=torch.sum(idx)/idx.shape[0]

    return new_features, prop_gaming

def agent_loss(w, theta, x):
    return theta - w + gaming_cost(theta, x)

def report_results(df, name):

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(36, 36))
    sns.set_context("poster")
    for i, metric in enumerate(["agent_loss_mean", "policy_loss_mean", "prop_gaming", "dist_mean", "slope", "intercept"]):
        x = int(i/3)
        y = i%3
        sns.lineplot(data=df, x="iteration", y=metric, ax=ax[x, y])
    plt.savefig("figs/{}.pdf".format(name))
    plt.close()


# ## Algorithm
# 1. Decision-maker announces rule $\beta^{t}$ and $E_{y\sim p^{t-1}}[y]$.
# 2. Agents chooses which features to play $x^{t}$.
# 3. $x^{t}$'s form the new population distribution $p^{t}.$
# 4. Decision-maker gives out allocations according to $\beta^{t}$ and $p^{t}$. Agents incur loss.
# 5. Decision-maker creates an updated rule $\beta^{t+1}.$

@argh.arg("--seed", default=0)
@argh.arg("--T", default=5)
@argh.arg("--game_weight", default=1.)
@argh.arg("--h", default=0.001)
@argh.arg("--eta", default=0.01)
@argh.arg("--n", default=10000)
@argh.arg("--exp_name", default="sim")

def play_game(n=10000, T=5000, game_weight=1., seed=0., h=0.001, eta=0.01, exp_name):

    torch.manual_seed(seed)

    theta = torch.rand(n, 1)
    theta_tilde = torch.clamp(theta + 0.5)


    agent_losses = torch.zeros(T, n)
    policy_losses = torch.zeros(T, n)
    betas = torch.zeros(T, 2)
    dists = []
    prop_gaming = []

    x = theta.copy()
    dist_mean = get_dist_mean(theta, beta)
    budget = dist_mean.clone()
    l_dics = {}
    for i in tqdm(range(T)):
        dists.append(dist_mean.item())
        betas[i] = beta.clone()
        perturbation = torch.bernoulli(torch.ones(n, 2) * 0.5)
        perturbation[perturbation==0.] = -1
        perturbation = perturbation * h
        beta_perturbed = beta.clone().repeat(n, 1) + perturbation
        x, p_game = get_agent_features(theta, theta_tilde, beta_perturbed, dist_mean, budget)
        prop_gaming.append(p_game)
    
        #compute dist mean
        dist_mean = get_dist_mean(x, beta)
        w = allocation(x, beta, dist_mean, budget)
        agent_losses[i] = agent_loss(w, theta, x).flatten()
        p_loss = policy_loss(w, theta)
        policy_losses[i] = p_loss.flatten()
        #gamma = torch.matmul(perturbation.T, p_loss)/(torch.matmul(perturbation.T, perturbation))
        gamma, _ = torch.solve(torch.matmul(perturbation.T, p_loss), torch.matmul(perturbation.T, perturbation)) 
        beta -= eta * gamma.flatten()
        dic = {"n":n,
               "T":T,
               "game_weight": game_weight,
               "seed": seed,
               "iteration": i,
               "agent_loss_mean": torch.mean(agent_losses, axis=1).item(),
               "agent_loss_std": torch.std(agent_losses, axis=1).item(), 
               "policy_loss_mean": torch.mean(policy_losses, axis=1).item(),
               "policy_loss_std": torch.std(policy_losses, axis=1).item(), 
               "prop_gaming": prop_gaming.item(),
               "dist_mean": dist_mean.item(), 
               "slope": beta[:, 0].item(),
               "intercept": betas[:, 1].item()}
        l_dics.append(dic)

    df = pd.DataFrame.from_dict(l_dics)
    df.to_csv("results/{}.csv".format(exp_name))
    report_results(df, exp_name)


