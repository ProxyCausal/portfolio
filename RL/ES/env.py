import pybullet_envs.bullet as bullet_envs

def make_env(env_name, render):
    env = None
    if env_name == 'minitaur':
        env = bullet_envs.MinitaurBulletEnv(render=render)
    elif env_name == 'racecar':
        env = bullet_envs.RacecarGymEnv(renders=render)
    return env