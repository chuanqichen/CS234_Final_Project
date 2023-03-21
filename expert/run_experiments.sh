CUDA_VISIBLE_DEVICES=0 python td3_with_obs.py --dirpath "test510" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 50,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=1 python td3_with_obs.py --dirpath "test511" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "test512" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "test513" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 &

CUDA_VISIBLE_DEVICES=0 python td3_with_obs.py --dirpath "test514" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=1 python td3_with_obs.py --dirpath "test515" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "test516" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "test517" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
