python run.py \
    --path_to_vm vmware_vm_data/Ubuntu0/Ubuntu0.vmx \
    --observation_type screenshot \
    --model claude-3-7-sonnet-20250219 \
    --test_all_meta_path evaluation_examples/test_all.json \
    --result_dir ./results \
    --action_space computer_13 \
    --sleep_after_execution 2.0 \
    --headless
