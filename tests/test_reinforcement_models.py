import os
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.reinforcement import train_ppo, train_dqn

# مسیر ذخیره‌سازی موقت برای تست
TEST_SAVE_DIR = "tests/saved_models/"
if not os.path.exists(TEST_SAVE_DIR):
    os.makedirs(TEST_SAVE_DIR)

# تست مدل PPO
def test_ppo_model():
    model_path = os.path.join(TEST_SAVE_DIR, "ppo_model.zip")
    model = train_ppo(
        env_id="CartPole-v1",  # محیط نمونه
        timesteps=1000,  # آموزش سریع برای تست
        save_path=model_path
    )
    assert os.path.exists(model_path), "مدل PPO ذخیره نشد!"
    print("تست مدل PPO با موفقیت انجام شد.")

# تست مدل DQN
def test_dqn_model():
    model_path = os.path.join(TEST_SAVE_DIR, "dqn_model.zip")
    model = train_dqn(
        env_id="CartPole-v1",  # محیط نمونه
        timesteps=1000,  # آموزش سریع برای تست
        save_path=model_path
    )
    assert os.path.exists(model_path), "مدل DQN ذخیره نشد!"
    print("تست مدل DQN با موفقیت انجام شد.")

if __name__ == "__main__":
    test_ppo_model()
    test_dqn_model()