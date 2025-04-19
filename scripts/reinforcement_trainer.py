import os
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.reinforcement import train_ppo, train_dqn

# مسیر ذخیره‌سازی مدل‌ها
SAVE_DIR = "saved_models/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def main():
    """
    آموزش مدل‌های PPO و DQN
    """
    # آموزش مدل PPO
    print("آموزش مدل PPO...")
    train_ppo(
        env_id="CartPole-v1",  # محیط نمونه برای آزمایش
        timesteps=50000,
        save_path=os.path.join(SAVE_DIR, "ppo_model.zip")
    )

    # آموزش مدل DQN
    print("آموزش مدل DQN...")
    train_dqn(
        env_id="CartPole-v1",  # محیط نمونه برای آزمایش
        timesteps=50000,
        save_path=os.path.join(SAVE_DIR, "dqn_model.zip")
    )

    print("آموزش مدل‌های یادگیری تقویتی به پایان رسید.")

if __name__ == "__main__":
    main()