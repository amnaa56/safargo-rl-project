
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SafarGo RL MVP", page_icon="🚗")

st.title("SafarGo: RL-Based Fleet Repositioning")
st.write(
    "A simulated demonstration comparing Q-Learning "
    "with a busiest-zone baseline."
)

st.info(
    "This is an educational simulation using generated data, "
    "not real SafarGo operational data."
)

if st.button("Run Simulation"):
    np.random.seed(42)

    zones = 40
    vehicles = 30
    episodes = 100

    demand = np.random.uniform(0.5, 3.0, zones)
    Q = np.zeros((2, zones, zones))
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    rewards = []

    for ep in range(episodes):
        positions = np.random.randint(0, zones, vehicles)
        total_reward = 0

        for t in range(24):
            state_time = 1 if 12 <= t < 18 else 0
            factor = 1.4 if state_time else 0.8
            requests = np.random.poisson(demand * factor)
            available = np.bincount(positions, minlength=zones)
            rides = int(np.minimum(requests, available).sum())
            unserved = int(np.maximum(requests - available, 0).sum())

            new_positions = positions.copy()

            for i, current in enumerate(positions):
                if np.random.random() < epsilon:
                    action = np.random.randint(zones)
                else:
                    action = int(np.argmax(Q[state_time, current]))

                fuel = 1.0 if action != current else 0.1
                reward = (
                    2 * demand[action] * factor
                    - 0.6 * fuel
                    + (0.5 if available[action] == 0 else 0)
                )

                next_time = 1 if 12 <= (t + 1) % 24 < 18 else 0
                old = Q[state_time, current, action]
                Q[state_time, current, action] = old + alpha * (
                    reward + gamma * np.max(Q[next_time, action]) - old
                )

                new_positions[i] = action

            total_reward += 2 * rides - 1.5 * unserved
            positions = new_positions

        rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.97)

    # Evaluation of the busiest-zone baseline and learned policy
    results = []

    for strategy in ["Busiest-zone baseline", "Q-Learning"]:
        rides_total = 0
        waiting_total = 0
        requests_total = 0
        fuel_total = 0

        for run in range(20):
            positions = np.random.randint(0, zones, vehicles)

            for t in range(24):
                factor = 1.4 if 12 <= t < 18 else 0.8
                requests = np.random.poisson(demand * factor)
                available = np.bincount(positions, minlength=zones)
                served = np.minimum(requests, available)

                rides = int(served.sum())
                unserved = int((requests - served).sum())

                rides_total += rides
                waiting_total += rides * 4 + unserved * 15
                requests_total += rides + unserved

                updated = positions.copy()

                for i, current in enumerate(positions):
                    if strategy == "Busiest-zone baseline":
                        action = int(np.argmax(demand))
                    else:
                        state_time = 1 if 12 <= t < 18 else 0
                        action = int(np.argmax(Q[state_time, current]))

                    fuel_total += 1 if action != current else 0.1
                    updated[i] = action

                positions = updated

        results.append({
            "Strategy": strategy,
            "Average wait (min)": waiting_total / max(requests_total, 1),
            "Completed rides": rides_total,
            "Fuel units": round(fuel_total, 1)
        })

    st.subheader("Training Reward Curve")
    fig, ax = plt.subplots()
    ax.plot(rewards, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Strategy Comparison")
    st.dataframe(pd.DataFrame(results).set_index("Strategy"))

    st.caption(
        "Results are simulated estimates. They are not real-world "
        "performance guarantees."
    )
