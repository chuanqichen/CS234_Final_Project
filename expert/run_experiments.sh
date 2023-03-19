#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python td3_with_obs.py --dirpath "test100" --filename "td3_fixed" --operation "both" --pi_arch 300, 200,100 --qf_arch 300,20o, 100 --max_timesteps 10000000
CUDA_VISIBLE_DEVICES=1 python td3_with_obs.py --dirpath "test200" --filename "td3_fixed" --operation "both" --pi_arch 300,100, 50 --qf_arch 100,20 --max_timesteps 10000000
CUDA_VISIBLE_DEVICES=2 python td3_with_obs.py --dirpath "test300" --filename "td3_fixed" --operation "both" --pi_arch 100,50,20--qf_arch 100,50,20 --max_timesteps 10000000
CUDA_VISIBLE_DEVICES=3 python td3_with_obs.py --dirpath "test400" --filename "td3_fixed" --operation "both" --pi_arch 50,20 --qf_arch 50,20 --max_timesteps 10000000
