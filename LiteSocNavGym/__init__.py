from gymnasium.envs.registration import register

register(
    id='LiteSocNavGym-v0',
    entry_point='LiteSocNavGym.envs:LiteSocNavGym',
)

register(
    id='DiscreteLiteSocNavGym-v0',
    entry_point='LiteSocNavGym.envs:DiscreteLiteSocNavGym',
)

