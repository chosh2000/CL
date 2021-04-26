OUTDIR=outputs/pMNIST/$(date +'%m-%d-%y_%Hh-%Mm')
REPEAT=1
mkdir -p $OUTDIR
# python -u ISG_pMNIST.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --num_task 100 --schedule 10 --batch_size_train 128   --rho 1 1  --lr 0.0001 --reglambda 0.01           | tee ${OUTDIR}/SIM_rho_10_10.log

# python -u ISG_pMNIST.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --num_task 100 --schedule 10 --batch_size_train 128   --rho 0.5 1  --lr 0.0001 --reglambda 0.01           | tee ${OUTDIR}/SIM_rho_5_10.log

# python -u ISG_pMNIST.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --num_task 100 --schedule 10 --batch_size_train 128   --rho 0.5 0.5  --lr 0.0001 --reglambda 0.01           | tee ${OUTDIR}/SIM_rho_5_5.log

#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.0001  --offline_training    | tee ${OUTDIR}/Offline.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.0001                        | tee ${OUTDIR}/Adam.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.01                          | tee ${OUTDIR}/SGD.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.001                         | tee ${OUTDIR}/Adagrad.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name EWC_online_mnist --lr 0.0001 --reg_coef 500   | tee ${OUTDIR}/EWC_online.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001            | tee ${OUTDIR}/Naive_Rehearsal_4000.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_16000  --lr 0.0001            | tee ${OUTDIR}/Naive_Rehearsal_16000.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name GEM_4000   --lr 0.1    --reg_coef 0.5         | tee ${OUTDIR}/GEM_4000.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation 10 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name GEM_16000  --lr 0.1    --reg_coef 0.5         | tee ${OUTDIR}/GEM_16000.log
# python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 25 --no_class_remap --force_out_dim 0 --schedule 10 --batch_size 128 --model_name MLP1000_drop --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.01        | tee ${OUTDIR}/MAS_drop.log

# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST  --method MAS  --model_type MLP  --mlp_size 100 --num_task 10 --print_freq 200 --schedule 10 --apply_SIM 1 --random_drop 0 --rho 0.5 0.5  --lr 0.0001 --reglambda 0.01  --alpha 1 --xi 0.01 --revert_head 1    | tee ${OUTDIR}/SIM_rho_5_5_revert1.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST  --method MAS  --model_type MLP  --mlp_size 100 --num_task 10 --print_freq 200 --schedule 10 --apply_SIM 1 --random_drop 0 --rho 0.5 0.5  --lr 0.0001 --reglambda 0.01  --alpha 1 --xi 0.01 --revert_head 0    | tee ${OUTDIR}/SIM_rho_5_5_revert0.log


# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method MAS  --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 1 --random_drop 1 --rho 0.2 0.2  --lr 0.0001 --reglambda 0.01  --alpha 1 --xi 0.01 --revert_head 0 --finetune_epoch 20   | tee ${OUTDIR}/SIM_rho22_reg001_random.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method MAS  --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 1 --random_drop 0 --rho 0.2 0.2  --lr 0.0001 --reglambda 0.01  --alpha 1 --xi 0.01 --revert_head 0 --finetune_epoch 20   | tee ${OUTDIR}/SIM_rho22_reg001.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method MAS  --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 0 --random_drop 0 --rho 1 1  --lr 0.0001 --reglambda 0     --revert_head 1 --finetune_epoch 20   | tee ${OUTDIR}/ANN_reg0.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method MAS  --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 0 --random_drop 0 --rho 1 1  --lr 0.0001 --reglambda 0.01  --revert_head 1 --finetune_epoch 20   | tee ${OUTDIR}/MAS_reg001.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method SI   --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 0 --random_drop 0 --rho 1 1  --lr 0.0001 --reglambda 0.8  --revert_head 1  --finetune_epoch 20   | tee ${OUTDIR}/SI_reg08.log
# sleep 5m
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method EWC  --model_type MLP  --mlp_size 2000 --num_task 100 --print_freq 200 --schedule 10 20 --apply_SIM 0 --random_drop 0 --rho 1 1  --lr 0.0001 --reglambda 5  --revert_head 1  --finetune_epoch 20   | tee ${OUTDIR}/EWC_reg5.log



# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset pMNIST   --method MAS  --model_type MLP  --mlp_size 2000 --num_task 10 --print_freq 200 --schedule 10 20 --apply_SIM 1 --random_drop 0 --rho 0.2 0.2  --lr 0.0001 --reglambda 0.01  --alpha 0.2 --xi 0.01 --revert_head 0 --finetune_epoch 20   | tee ${OUTDIR}/SIM_r22_a02_x001.log



