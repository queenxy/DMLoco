```
python script/run.py --config-name=pre_diffusion_unet --config-dir=cfg/aliengo/pretrain/multi_gait_and_vel
python script/eval_pretrain.py --config-name=eval_diffusion_unet --config-dir=cfg/aliengo/eval/multi_gait_and_vel
python script/run.py --config-name=ft_ppo_diffusion_unet --config-dir=cfg/aliengo/finetune/multi_gait_and_vel
```
