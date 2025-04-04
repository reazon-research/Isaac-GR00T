# OpenARM

# Custom configuration
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
