
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SafarGo RL Simulator",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 SafarGo: Smart Vehicle Repositioning")
st.write(
    "A simulation comparing Random Repositioning "
    "with a Q-Learning strategy."
)

st.info(
    "This educational MVP uses simulated demand, not real SafarGo data."
)

# Simulation settings
ZONES = 10
VEHICLES = 30
STEPS = 40
EPISODES = 150

alpha = 0.15
gamma = 0.90
epsilon = 0.25

# Reproducible simulated demand
rng = np.random.default_rng(42)
zone_demand = rng.integers(3, 16, size=ZONES)

# Demand changes over time
def get_demand(step):
    peak_zone = (step // 5) % ZONES
    demand = zone_demand.astype(float).copy()
    demand[peak_zone] += 10
    demand[(peak_zone + 1) % ZONES] += 5
    return demand

# State = current zone and demand category
def get_state(zone, demand):
    if demand < 7:
        level = 0
    elif demand < 13:
        level = 1
    else:
        level = 2
    return zone * 3 + level

N_STATES = ZONES * 3
N_ACTIONS = ZONES

# Training Q-Learning
if st.button("Train and Compare Strategies", type="primary"):
    q_table = np.zeros((N_STATES, N_ACTIONS))
    reward_history = []

    for episode in range(EPISODES):
        vehicle_zones = rng.integers(0, ZONES, size=VEHICLES)
        total_reward = 0

        for step in range(STEPS):
            demand = get_demand(step)
            new_positions = []

            for zone in vehicle_zones:
                state = get_state(zone, demand[zone])

                if rng.random() < epsilon:
                    action = rng.integers(0, ZONES)
                else:
                    action = int(np.argmax(q_table[state]))

                # Reward: serving demand is good; travel costs fuel
                rides = min(1, demand[action] / 10)
                fuel_cost = 0.15 if action == zone else 0.5
                reward = rides * 2 - fuel_cost

                next_state = get_state(action, demand[action])

                q_table[state, action] += alpha * (
                    reward
                    + gamma * np.max(q_table[next_state])
                    - q_table[state, action]
                )

                total_reward += reward
                new_positions.append(action)

            vehicle_zones = np.array(new_positions)

        reward_history.append(total_reward)

        # Gradually reduce random exploration
        epsilon = max(0.05, epsilon * 0.985)

    # Evaluation: same demand sequence for both policies
    eval_rng = np.random.default_rng(2026)
    random_wait = []
    q_wait = []
    random_rides = []
    q_rides = []
    random_fuel = []
    q_fuel = []

    for policy in ["Random", "Q-Learning"]:
        vehicle_zones = eval_rng.integers(
            0, ZONES, size=VEHICLES
        )

        waits = []
        rides_total = 0
        fuel_total = 0

        for step in range(STEPS):
            demand = get_demand(step)
            available = np.bincount(
                vehicle_zones, minlength=ZONES
            )
            served = np.minimum(available, demand)
            unmet = np.maximum(demand - available, 0)

            rides_total += served.sum()
            waits.append(unmet.mean())

            next_zones = []

            for zone in vehicle_zones:
                if policy == "Random":
                    action = eval_rng.integers(0, ZONES)
                else:
                    state = get_state(zone, demand[zone])
                    action = int(np.argmax(q_table[state]))

                fuel_total += 1 if action != zone else 0.2
                next_zones.append(action)

            vehicle_zones = np.array(next_zones)

        if policy == "Random":
            random_wait = waits
            random_rides = rides_total
            random_fuel = fuel_total
        else:
            q_wait = waits
            q_rides = rides_total
            q_fuel = fuel_total

    results = pd.DataFrame({
        "Strategy": ["Random Baseline", "Q-Learning"],
        "Average Unmet Demand": [
            np.mean(random_wait), np.mean(q_wait)
        ],
        "Completed Rides (simulated)": [
            random_rides, q_rides
        ],
        "Fuel Units (simulated)": [
            round(random_fuel, 1), round(q_fuel, 1)
        ]
    })

    st.subheader("Training Reward Curve")
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Training Reward")
    ax.set_title("Q-Learning Training Progress")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Strategy Comparison")
    st.dataframe(results, use_container_width=True)

    st.caption(
        "Results are simulation estimates, not guaranteed real-world "
        "performance. A lower unmet-demand value is better. "
        "Q-Learning is not guaranteed to outperform the random baseline."
    )
