import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import random
from collections import deque
import time

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create training environment with no rendering
env = gym.make('LunarLander-v3', continuous=False, gravity=-11.9,
               enable_wind=False, wind_power=9.0, turbulence_power=4)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
BUFFER_SIZE = int(1e5)
TAU = 1e-3
LEARNING_RATE = 1 #1
UPDATE_EVERY = 4
NUM_EPISODES =  501
DISPLAY_EVERY = 100  # Show AI play every 10 episodes

# Get environment dimensions
num_actions = env.action_space.n
state_size = env.observation_space.shape[0]


class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, num_actions):
        self.state_size = state_size
        self.num_actions = num_actions

        self.qnetwork_local = DQN(state_size, num_actions).to(DEVICE)
        self.qnetwork_target = DQN(state_size, num_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_actions))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(TAU)

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def human_play(ai_score):
    pygame.init()
    pygame.display.init()

    # Create instruction window
    info_window = pygame.display.set_mode((500, 400))
    pygame.display.set_caption("Human Play")
    info_window.fill((0, 0, 0))

    font = pygame.font.Font(None, 36)
    text1 = font.render(f"AI Score: {ai_score:.2f}", True, (255, 255, 255))
    text2 = font.render("Controls:", True, (255, 255, 255))
    text3 = font.render("↑: Main Engine", True, (255, 255, 255))
    text4 = font.render("←/→: Side Thrusters", True, (255, 255, 255))
    text5 = font.render("Press SPACE to Start", True, (255, 255, 255))

    info_window.blit(text1, (50, 50))
    info_window.blit(text2, (50, 120))
    info_window.blit(text3, (50, 170))
    info_window.blit(text4, (50, 220))
    info_window.blit(text5, (50, 300))

    pygame.display.flip()

    # Wait for SPACE to start
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return 0

    pygame.display.quit()

    # Start the game
    human_env = gym.make('LunarLander-v3', continuous=False, gravity=-11.9,
               enable_wind=False, wind_power=9, turbulence_power=4, render_mode='human')
    clock = pygame.time.Clock()
    observation, info = human_env.reset()
    total_reward = 0
    done = False

    while not done:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 3
        elif keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_ESCAPE]:
            done = True

        observation, reward, terminated, truncated, info = human_env.step(action)
        total_reward += reward
        done = done or terminated or truncated

        # Draw score comparison during game
        pygame.display.set_caption(f"Your Score: {total_reward:.2f} | AI Score: {ai_score:.2f}")
        clock.tick(60)

    # After crash/landing, show final score comparison
    pygame.init()
    score_window = pygame.display.set_mode((500, 400))
    pygame.display.set_caption("Game Over - Press SPACE to continue")
    score_window.fill((0, 0, 0))

    # Create and render final score text
    font = pygame.font.Font(None, 36)
    final_text1 = font.render("Game Over!", True, (255, 255, 255))
    final_text2 = font.render(f"Your Score: {total_reward:.2f}", True, (255, 255, 255))
    final_text3 = font.render(f"AI Score: {ai_score:.2f}", True, (255, 255, 255))
    score_diff = total_reward - ai_score
    color = (0, 255, 0) if score_diff > 0 else (255, 0, 0)  # Green if beat AI, red if not
    final_text4 = font.render(f"Difference: {score_diff:.2f}", True, color)
    final_text5 = font.render("Press SPACE to continue", True, (255, 255, 255))

    # Position text
    score_window.blit(final_text1, (180, 50))
    score_window.blit(final_text2, (150, 120))
    score_window.blit(final_text3, (150, 170))
    score_window.blit(final_text4, (150, 220))
    score_window.blit(final_text5, (120, 300))

    pygame.display.flip()

    # Wait for SPACE to continue
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    waiting = False

    # Cleanup
    human_env.close()
    pygame.quit()
    return total_reward


def main():
    # Initialize agent
    agent = Agent(state_size, num_actions)

    # Lists to store scores
    scores = []
    scores_window = deque(maxlen=100)
    eps = EPS_START

    try:
        print(f"Starting training on device: {DEVICE}")
        print("Training silently... Will show AI play every", DISPLAY_EVERY, "episodes")

        for episode in range(1, NUM_EPISODES + 1):
            # Close previous environment if exists
            if 'env' in locals():
                env.close()

            # Determine if this episode should be rendered
            render = episode % DISPLAY_EVERY == 0

            # Create appropriate environment
            if render:
                env = gym.make('LunarLander-v3', continuous=False, gravity=-11.9,
               enable_wind=False, wind_power=9.0, turbulence_power=4, render_mode='human')
                print(f"\nRendering Episode {episode}")
            else:
                env = gym.make('LunarLander-v3', continuous=False, gravity=-11.9,
               enable_wind=False, wind_power=9.0, turbulence_power=4)

            state, _ = env.reset()
            score = 0

            for t in range(1000):  # max steps per episode
                action = agent.act(state, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            eps = max(EPS_END, EPS_DECAY * eps)

            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")

            if render:
                ai_score = score
                print(f"\nAI Score: {ai_score:.2f}")
                print("\nYour turn! Try to beat the AI's score!")
                human_score = human_play(ai_score)
                print(f"\nScore Comparison:")
                print(f"AI Score: {ai_score:.2f}")
                print(f"Your Score: {human_score:.2f}")
                print(f"Difference: {human_score - ai_score:.2f}")

            if episode % 100 == 0:
                print(f'\nEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}')

            if np.mean(scores_window) >= 200.0:
                print(f'\nEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        env.close()


if __name__ == "__main__":
    main()

