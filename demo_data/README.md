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

The inference server was set up on a DGX cluster using port forwarding
```sh
ssh -L 8888:localhost:888 <address-to-cluster>
```

On the local host, verify that the port is listening
```sh
netstat --listening |grep 8888
```



