import torch

class AgentType:

    def __init__(self, natural_feature_dist, gamed_feature_dist, manipulation_dist):
        self.natural_feature_dist = natural_feature_dist
        self.gamed_feature_dist = gamed_feature_dist
        self.manipulation_dist = manipulation_dist
    

class Agent:
    def __init__(self, agent_type):
        self.agent_type = agent_type
        self.a = self.agent_type.natural_feature_dist.sample()
        self.b = self.agent_type.gamed_feature_dist.sample()
        self.gamma = self.agent_type.manipulation_dist.sample()

        
        self.cumulative_loss_a = [0.+torch.randn(1) *0.05]
        self.cumulative_loss_b = [self.gamma + torch.randn(1)*0.05]
        self.allocations = []
        self.actions = []

    def update_reward(self, allocation):
        noise = torch.randn(1) * 0.05
        self.allocations.append(allocation)
        if self.actions[-1]:
            self.cumulative_loss_a.append(-allocation+noise)
        else:
            self.cumulative_loss_b.append(-allocation + self.gamma+noise)

    def present_feature(self):
        rand_num = torch.rand(1)
        if rand_num > 0.1:
            if torch.Tensor(self.cumulative_loss_a).mean() < torch.Tensor(self.cumulative_loss_b).mean():
                self.actions.append(True)
                return self.a
            else:
                self.actions.append(False)
                return self.b
        else:
            if rand_num < 0.05:
                self.actions.append(True)
                return self.a
            else:
                self.actions.append(False)
                return self.b



    def outcome(self):
        outcome = abs(self.allocations[-1] - self.a)
        assert outcome < 1
        return outcome
