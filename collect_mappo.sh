TASK=vmas/navigation

for s in 90 91 92 93 94 95 96 97 98 99 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004; do
  python benchmarl/collect.py algorithm=mappo task=$TASK seed=$s task.collisions=True task.n_agents=2 task.agent_radius=0.1 task.lidar_range=0.35 \
    ++collect.policy=reference \
    ++collect.total_frames=100000 \
    ++collect.output_dir=dataset/mappo_2agent_seed_${s}
done
