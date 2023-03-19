CUDA_VISIBLE_DEVICES=0 python td3_with_obs.py --dirpath "test410" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 50,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=1 python td3_with_obs.py --dirpath "test411" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "test412" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,20 --max_timesteps 10000000 --placement random --discount 0.98 &
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "test413" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement random --discount 0.98 &

CUDA_VISIBLE_DEVICES=0 python td3_with_obs.py --dirpath "test414" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=1 python td3_with_obs.py --dirpath "test415" --filename "td3" --operation "both" --pi_arch 50,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "test416" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "test417" --filename "td3" --operation "both" --pi_arch 100,20 --qf_arch 100,50,20 --max_timesteps 10000000 --placement fixed --discount 0.98 &
