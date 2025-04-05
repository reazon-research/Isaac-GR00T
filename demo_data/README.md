# OpenArm

# Fine-tuning
- DOF 1-7 for arm
- DOF 8 for gripper

- copy or symlink to data folder, e.g.
```sh
ln -s <path-to-data> $PWD/openarm.PickNPlace/data 
ln -s <path-to-videos> $PWD/openarm.PickNPlace/videos
```

```sh
python3 scripts/gr00t_finetune.py --dataset-path ./demo_data/openarm.PickNPlace/ \
--num-gpus 4 \
--output-dir ~/openarm-pnp-checkpoints \
--max-steps 2000 --data-config \
openarm --video_backend \
torchvision_av --batch_size 64
```

The `annotation.human.action.task_description` property needs to be kept consistent for tasks, e.g. 0

## Inference

The host computer was run on a 3090TI (24GB VRAM)

The port 8888 was exposed to the local subnet, e.g. 192.168.111.0
```sh
sudo ufw allow from <local-network-ip>/24 to any port 8888 proto tcp
```


