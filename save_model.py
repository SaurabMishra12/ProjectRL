def save_model(self, filepath: str):
        """Save the actor and critic MLP weights to a .npz file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath,
                 actor_W1=self.actor.W1, actor_b1=self.actor.b1,
                 actor_W2=self.actor.W2, actor_b2=self.actor.b2,
                 actor_W3=self.actor.W3, actor_b3=self.actor.b3,
                 critic_W1=self.critic.W1, critic_b1=self.critic.b1,
                 critic_W2=self.critic.W2, critic_b2=self.critic.b2,
                 critic_W3=self.critic.W3, critic_b3=self.critic.b3,
                 # Save mean_dim so load_model can validate architecture compatibility
                 mean_dim=np.array(self.mean_dim))

    def load_model(self, filepath: str):
        """Load the actor and critic MLP weights from a .npz file."""
        data = np.load(filepath)
        self.actor.W1 = data['actor_W1']
        self.actor.b1 = data['actor_b1']
        self.actor.W2 = data['actor_W2']
        self.actor.b2 = data['actor_b2']
        self.actor.W3 = data['actor_W3']
        self.actor.b3 = data['actor_b3']
        self.critic.W1 = data['critic_W1']
        self.critic.b1 = data['critic_b1']
        self.critic.W2 = data['critic_W2']
        self.critic.b2 = data['critic_b2']
        self.critic.W3 = data['critic_W3']
        self.critic.b3 = data['critic_b3']

"""
Dataset Loaders: GSM8K, HotpotQA, MATH (AQuA), TruthfulQA, and MBPP.
Returns STRICTLY separated (train, test) tuples to prevent data leakage.
"""



logger = setup_logging()


def _generate_synthetic_samples(n_train: int, n_test: int, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Generate synthetic GSM8K-style math word problems for testing.
    Returns strictly disjoint train and test sets.
    """
    rng = random.Random(seed)
    templates = [
        {
            "template": "{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} hav