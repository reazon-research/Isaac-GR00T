## NVIDIA Isaac GR00T N1

For deployment on [OpenArm](https://github.com/reazon-research/openarm)

https://github.com/user-attachments/assets/ff1c8d33-7455-4cbe-ba84-98aa97ec9dc0


## Installation
```sh
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

## Known Issues

- Fine-tuning image inputs requires significant VRAM (more than 24GB): [reduce parallel processes in groot_finetune.py](https://github.com/NVIDIA/Isaac-GR00T/issues/24#issuecomment-2757807812)

## Results

![openarm-1-5k-steps.png](./media/openarm-1-5k-steps.png)

## License 

```
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```
