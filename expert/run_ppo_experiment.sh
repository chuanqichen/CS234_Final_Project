CUDA_VISIBLE_DEVICES=0 python ppo_with_obs.py --dirpath "small_obs_ppo_100_50_20_1e4_clip_07" --filename "ppo" --operation "both" --pi 100,50,20 --vf 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 --learning_rate 0.0001 --clip_range 0.4 &