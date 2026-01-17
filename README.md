```
$ python demo3/evaluate.py task=ms-stack-cube-semi checkpoint=/path/to/stack-cube.pt save_video=true save_trajectory=true obs="state" obs_save="rgb"
$ python demo3/evaluate.py task=ms-humanoid-transport-box-semi checkpoint=/path/to/humanoid-transport-box.pt save_video=true save_trajectory=true obs="state" obs_save="rgb"
```


```
$ python demo3/train.py task=ms-peg-insertion-semi steps=5000000 demo_path=/data/sunhaowen/hw_mine/sil_demo/ms-peg-insertion-semi_trajectories_10.pkl enable_reward_learning=true

$ python demo3/train.py task=ms-stack-cube-semi steps=5000000 demo_path=/data/sunhaowen/hw_mine/sil_demo/ms-stack-cube-semi_trajectories_10.pkl enable_reward_learning=true

$ python demo3/train.py task=ms-pick-place-semi steps=5000000 demo_path=/data/sunhaowen/hw_mine/sil_demo/ms-pick-place-semi_trajectories_10.pkl enable_reward_learning=true

$ python demo3/train.py task=mw-assembly-semi steps=500000 obs=rgb demo_path=/path/to/mw-demos/assembly-200.pkl enable_reward_learning=true
$ python demo3/train.py task=ms-peg-insertion-semi steps=500000 obs=rgb demo_path=/data2/user/sunhaowen/hw_mine/intrinsic_demo3/ms-peg-insertion-semi_trajectories_10.pkl enable_reward_learning=true
```


```
$ python demo3/train.py task=ms-stack-cube-semi steps=1000000 enable_reward_learning=false demo_sampling_ratio=0.0 policy_pretraining=False
```

```
conda env create -f docker/environment.yaml
```
