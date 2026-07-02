import math
import copy
import numpy as np
import torch

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=25, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = next(model.parameters()).device

    def get_action_prob(self, env, temperature=1.0):
        root = Node(prior=1.0)
        self._expand(root, env)

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(env)
            search_path = [node]

            # 1. Selection
            while len(node.children) > 0:
                action, node = self._select_child(node)
                _, reward, terminal, _ = sim_env.step(action)
                search_path.append(node)
                if terminal:
                    break

            # 2. Expansion & Evaluation
            if not terminal:
                value = self._expand(node, sim_env)
            else:
                # If terminal, reward is from the perspective of the player who just moved
                # So the value for the CURRENT player (who has no moves) is -reward
                value = -reward

            # 3. Backpropagation
            self._backpropagate(search_path, value)

        # Calculate action probabilities based on visit counts
        action_visits = {a: child.visit_count for a, child in root.children.items()}
        actions = list(action_visits.keys())
        counts = list(action_visits.values())
        
        if temperature == 0:
            best_action = actions[np.argmax(counts)]
            probs = np.zeros(209)
            probs[best_action] = 1.0
            return probs

        counts = np.array(counts) ** (1.0 / temperature)
        probs = counts / np.sum(counts)
        
        full_probs = np.zeros(209)
        for a, p in zip(actions, probs):
            full_probs[a] = p
            
        return full_probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            # UCB formula (AlphaZero style)
            # We invert child.value() because it's a 2-player zero-sum game
            q_value = -child.value()
            u_value = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, env):
        obs = torch.FloatTensor(env.get_observation()).unsqueeze(0).to(self.device)
        mask = env.get_legal_action_mask()
        
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs, torch.BoolTensor(mask).unsqueeze(0).to(self.device))
            logits = logits.squeeze(0).cpu().numpy()
            value = value.item()
            
        # Softmax over legal actions
        exp_logits = np.exp(logits - np.max(logits)) # stability
        probs = exp_logits * mask
        probs /= np.sum(probs)

        # Add legal actions as children
        legal_actions = np.where(mask)[0]
        for action in legal_actions:
            node.children[action] = Node(prior=probs[action])
            
        return value

    def _backpropagate(self, search_path, value):
        # We invert the value at each step up the tree because players alternate
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value 
