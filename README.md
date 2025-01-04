# 🤖 AILIVE Machine Learning Node

Welcome to the **Machine Learning Node** repository! This project demonstrates a simplified machine learning pipeline that will eventually transition to **Google Cloud's Vertex AI Platform** for scalability. By making this code public, we aim to provide transparency into the inner workings of our machine learning agents during their training process. 🌐

---

## Overview

This repository showcases the training of our machine learning agents with a focus on clarity and simplicity. While currently limited to a few files, the project highlights key elements of agent training and observation exporting.

### Current Demonstration

- **Agent:** Zero 🦾  
- **Task:** Learning how to walk 🚶  
- **Implementation:** Hardcoded for demonstration purposes to keep the process straightforward and easily understandable.  

This approach allows anyone to view the source code and gain insights into how our agents learn and improve over time.

---

## 🚀 Future Plans

- **Scalability:** Transitioning the node to **Google Cloud** to handle larger workloads and multiple agents simultaneously.  
- **Flexibility:** Expanding beyond the hardcoded "walking" skill to allow for dynamic skill assignments.  
- **Improved Transparency:** Enhancing monitoring and visualization tools to better showcase agent progress.  

---

## Session Data Export

The `session.php` script is a key component of our system. It exports session data, which is used for internal communication between our web servers. This provides real-time updates and synchronization of agent training progress.

You can access the exported session data publicly via the following endpoint:  
**[https://api.ailive.co/v1/sessions/zero/walking](https://api.ailive.co/v1/sessions/zero/walking)**

This endpoint provides a JSON response containing detailed session information, enabling you to monitor agent progress and explore the training data.

---

## Why Make It Public?

We believe in the value of transparency and collaboration. By sharing this repository:
- You can explore how the training pipeline works.  
- Gain insights into machine learning techniques applied in reinforcement learning.  
- Contribute suggestions or improvements to the codebase.  

---

## How It Works ⚙️

1. **Agent Training**: The agent (`Zero`) is trained in a simulated environment to learn specific tasks (e.g., walking).  
2. **Observation Exporting**: Training data is exported at regular intervals to monitor the agent's performance.  
3. **Model Saving**: The agent's progress is saved, allowing for resumption or analysis at any point.  