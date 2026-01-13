TASK=vmas/navigation

for s in 991 992 993 994 995; do
  python benchmarl/collect.py algorithm=mappo task=$TASK seed=$s task.collisions=True task.n_agents=1 task.agent_radius=0.1 task.lidar_range=0.35 \
    ++collect.policy=reference \
    ++collect.total_frames=100000 \
    ++collect.output_dir=dataset/mappo_seed${s}
done
