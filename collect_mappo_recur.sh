TASK=vmas/navigation

for s in 991 992 993 994 995 996 997 998 999; do
  python benchmarl/collect.py algorithm=mappo task=$TASK seed=$s task.collisions=True task.n_agents=2 task.agent_radius=0.1 task.lidar_range=0.35 \
    ++collect.policy=reference \
    ++collect.total_frames=100000 \
    ++collect.output_dir=dataset/mappo_2agent_recurrent_seed_${s} \
    ++collect.dataset_format=flat_with_history \
    ++collect.max_history_length=8 \
    ++collect.max_future_length=4
done

