envs = {
        #from minitaur_gym_env._get_observation
        'minitaur': {
            'NAME': 'MinitaurBulletEnv-v0',
            'INPUT_DIM': 28, #angles, velocities, torques for each of 8 motors + 4 quaternion coordinates
            'HIDDEN_DIMS': [64, 32],
            'ACTION_DIM': 8, #angle for each motor
            'ACTIVATION': 'tanh',
            'MAX_FRAMES': 1000 #what the reward function is based on
        },
        'racecar':  {
            'NAME': 'RacecarBulletEnv-v0',
            'INPUT_DIM': 2,
            'HIDDEN_DIMS': [20, 20],
            'ACTION_DIM': 2,
            'ACTIVATION': 'none',
            'MAX_FRAMES': 1000
        }
}
