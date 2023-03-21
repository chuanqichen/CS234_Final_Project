CUDA_VISIBLE_DEVICES=0 python ppo_with_obs.py --dirpath "small_obs_ppo_50_20_1e4_clip_04" --filename "ppo" --operation "both" --pi 50,20 --vf 50,20 --max_timesteps 10000000 --placement random --discount 0.98 --learning_rate 0.0001 --clip_range 0.4 &
CUDA_VISIBLE_DEVICES=1 python ppo_with_obs.py --dirpath "small_obs_ppo_100_50_20_1e4_clip_04" --filename "ppo" --operation "both" --pi 100,50,20 --vf 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 --learning_rate 0.0001 --clip_range 0.4 &
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "small_obs_td3_50_20_1e3" --filename "td3" --operation "both" --pi 50,20 --vf 50,20 --max_timesteps 10000000 --placement random --discount 0.98 --learning_rate 0.001 &
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "small_obs_td3_100_50_20_1e3" --filename "td3" --operation "both" --pi 100,50,20 --vf 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 --learning_rate 0.001 &