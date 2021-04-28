OUTDIR=outputs/sCIFAR/$(date +'%m-%d-%y_%H:%M')
REPEAT=1
mkdir -p $OUTDIR

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 0.5 	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho155_o1_a5.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 1 	--reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 1 1   	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_a5.log


# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1 	   --reglambda 0 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg0_a5.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1_a5.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 1 	--alpha 1    | tee ${OUTDIR}/SIM_rho111_reg1_a1.log


# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho151_reg1_random.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho144_reg1_random1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho144_reg1_random2.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.7 0.7   --reglambda 1 --alpha 0.8	| tee ${OUTDIR}/SIM_rho177_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.6 0.6   --reglambda 1 --alpha 0.8	| tee ${OUTDIR}/SIM_rho166_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.3 0.3   --reglambda 1 --alpha 1	| tee ${OUTDIR}/SIM_rho133_reg1_a1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.2 0.2   --reglambda 1 --alpha 1	| tee ${OUTDIR}/SIM_rho122_reg1_a1.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.4 1 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho141_reg1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 1 0.4 --reglambda 1	| tee ${OUTDIR}/SIM_randomeven_rho114_reg1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.5 0.5 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho155_reg1.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.5 1 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho151_reg1.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.12    | tee ${OUTDIR}/SIM_rho144_reg1_a12.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.14    | tee ${OUTDIR}/SIM_rho144_reg1_a14.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.16    | tee ${OUTDIR}/SIM_rho144_reg1_a16.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.18    | tee ${OUTDIR}/SIM_rho144_reg1_a18.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.20    | tee ${OUTDIR}/SIM_rho144_reg1_a20.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.22    | tee ${OUTDIR}/SIM_rho144_reg1_a22.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.24    | tee ${OUTDIR}/SIM_rho144_reg1_a24.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.26    | tee ${OUTDIR}/SIM_rho144_reg1_a26.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --random_drop 1 --rho 1 1 0.4   --reglambda 1 --alpha 0	    | tee ${OUTDIR}/SIM_rho114_reg1_random_dropout.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --random_drop 1 --rho 1 0.4 1   --reglambda 1 --alpha 0	    | tee ${OUTDIR}/SIM_rho141_reg1_random_dropout.log


# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --reglambda 1 --alpha 0.4	| tee ${OUTDIR}/SIM_rho144_reg1_a04_dropout.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --reglambda 1 --alpha 0.8	| tee ${OUTDIR}/SIM_rho144_reg1_a08_dropout.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 1 --out_dir $OUTDIR  --schedule 50 80  --random_drop 1 --rho 1 1 1     --reglambda 1 	    | tee ${OUTDIR}/SIM_rho111_reg1_MAS_batchnorm.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --random_drop 1 --rho 1 0.4 0.4 --reglambda 1 	    | tee ${OUTDIR}/SIM_rho144_reg1_random_batchnorm.lo
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.4 0.4 --reglambda 1	    | tee ${OUTDIR}/SIM_rho144_reg1_randomeven_batchnorm.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --reglambda 1 --alpha 0.2	| tee ${OUTDIR}/SIM_rho144_reg1_a02_batchnorm.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --reglambda 1 --alpha 0.6	| tee ${OUTDIR}/SIM_rho144_reg1_a06_batchnorm.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --reglambda 1 --alpha 1.0	| tee ${OUTDIR}/SIM_rho144_reg1_a1_batchnorm.log

# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.0 --reglambda 1 --alpha 0.0 --beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib00.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.1 --reglambda 1 --alpha 0.0	--beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib01.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.2 --reglambda 1 --alpha 0.0	--beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib02.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.3 --reglambda 1 --alpha 0.0	--beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib03.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.4 --reglambda 1 --alpha 0.0	--beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib04.log
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --init_model 0 --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.4 0.4 --inhib 0.5 --reglambda 1 --alpha 0.0	--beta 0.9	| tee ${OUTDIR}/SIM_rho144_reg1_a0_b09_dropout_inhib05.log



# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method MAS  --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 1 --rho 1 0.4 0.4 --reglambda 0.1 --alpha 1 --xi 1| tee ${OUTDIR}/SIM_rho144_reg01_a0_dropout.log    
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method MAS  --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 1 --rho 1 0.4 0.4 --reglambda 0.1 --alpha 1 --xi 1| tee ${OUTDIR}/SIM_rho144_reg01_a0_dropout.log    
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method MAS  --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 0 --rho 1 1 1     --reglambda 0.1   | tee ${OUTDIR}/MAS_reg01_dropout.log    
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method EWC  --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 0 --rho 1 1 1     --reglambda 100   | tee ${OUTDIR}/EWC_reg100_dropout.log    
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method SI   --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 0 --rho 1 1 1     --reglambda 10    | tee ${OUTDIR}/SI_reg10_dropout.log    
# python -u ISG_sCIFAR.py --use_gpu 1 --repeat $REPEAT --method MAS  --dataset sCIFAR100 --init_model 0 --out_dir $OUTDIR  --schedule 20 40  --apply_SIM 0 --rho 1 1 1     --reglambda 0     | tee ${OUTDIR}/ANN_reg0_dropout.log    

# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN  --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 1 0.4 0.4 --reglambda 1 --random_drop 1 --revert_head 0| tee ${OUTDIR}/SIM_rho11144_reg1_random.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1     --reglambda 0 	--random_drop 0 --revert_head 1| tee ${OUTDIR}/ANN_rho111_size2_reg0.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1     --reglambda 1 	--random_drop 0 --revert_head 1| tee ${OUTDIR}/MAS_rho111_size2_reg1.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method SI   --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1     --reglambda 80 	--random_drop 0 --revert_head 1| tee ${OUTDIR}/SI_rho111_size2_reg80.log
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method EWC  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1     --reglambda 1000 	--random_drop 0 --revert_head 1| tee ${OUTDIR}/EWC_rho111_reg1000.log

# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 1  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.8 0.8 0.8 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11888_size2_reg1_a5_xi001.log 
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.6 0.6 0.6 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11666_size2_reg1_a5_xi001.log 
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 0.4 0.4 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11444_size2_reg1_a5_xi001.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 1 0.4   --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11414_size2_reg1_a5_xi001.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 1 0.4 0.4   --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0 --random_drop 1| tee ${OUTDIR}/SIM_rho11144_size2_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 1 0.4   --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0 --random_drop 1| tee ${OUTDIR}/SIM_rho11414_size2_reg1_random.log    
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 1 --init_model 1  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1       --reglambda 0 	--random_drop 0 --revert_head     1| tee ${OUTDIR}/ANN_rho11111_size1_reg0.log
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 1 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1       --reglambda 1 	--random_drop 0 --revert_head     1| tee ${OUTDIR}/MAS_rho11111_size1_reg1.log
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 1 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 1 0.4 0.4   --reglambda 1   --alpha 5 --xi 0.01 --revert_head 1| tee ${OUTDIR}/SIM_rho11144_size1_reg1_a5_xi001.log    
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method SI   --model_type CNN --cnn_size 1 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1       --reglambda 80 	--random_drop 0 --revert_head     1| tee ${OUTDIR}/SI__rho11111_size1_reg80.log
python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method EWC  --model_type CNN --cnn_size 1 --init_model 0  --schedule 20 40  --apply_SIM 0 --rho 1 1 1 1 1       --reglambda 180 --random_drop 0 --revert_head     1| tee ${OUTDIR}/EWC_rho11111_size1_reg1000.log

# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 1 0.4   --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0 --random_drop 1| tee ${OUTDIR}/SIM_rho11414_size2_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.8 0.8 0.8 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11888_size2_reg1_a5_xi001.log 
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.6 0.6 0.6 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11666_size2_reg1_a5_xi001.log 
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 0.4 0.4 --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11444_size2_reg1_a5_xi001.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 0.4 1 0.4   --reglambda 1 --alpha 5 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11414_size2_reg1_a5_xi001.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 1 1 1 0.2 0.2   --reglambda 1 --alpha 0 --xi 0.01 --revert_head 0| tee ${OUTDIR}/SIM_rho11122_size4_reg1_a0_xi001.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.2 0.2 0.2 0.2 0.2   --reglambda 1 --random_drop 1   --revert_head 0| tee ${OUTDIR}/SIM_rho22222_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.3 0.3 0.3 0.3 0.3   --reglambda 1 --random_drop 1   --revert_head 0| tee ${OUTDIR}/SIM_rho33333_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.4 0.4 0.4 0.4 0.4   --reglambda 1 --random_drop 1   --revert_head 0| tee ${OUTDIR}/SIM_rho44444_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.5 0.5 0.5 0.5 0.5   --reglambda 1 --random_drop 1   --revert_head 0| tee ${OUTDIR}/SIM_rho55555_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.6 0.6 0.6 0.6 0.6   --reglambda 1 --random_drop 1   --revert_head 1| tee ${OUTDIR}/SIM_rho66666_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.7 0.7 0.7 0.7 0.7   --reglambda 1 --random_drop 1   --revert_head 1| tee ${OUTDIR}/SIM_rho77777_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 4 --init_model 0  --schedule 20 40  --apply_SIM 1 --rho 0.6 0.6 0.6 0.6 0.6   --reglambda 1 --alpha 0 --xi 0.01   --revert_head 1| tee ${OUTDIR}/SIM_rho66666_size4_reg1_random.log    
# python -u SIM.py --use_gpu 1 --repeat $REPEAT --out_dir $OUTDIR  --dataset sCIFAR100  --method MAS  --model_type CNN --cnn_size 2 --init_model 1  --schedule 20 40  --apply_SIM 1 --rho 0.7 0.7 0.7 0.7 0.7   --reglambda 1 --alpha 0 --xi 0.01   --revert_head 1| tee ${OUTDIR}/SIM_rho77777_size2_reg1_random.log    
#Reglambda used for resnet, CIFAR100
#EWC: 100
#EWC_online: 3000
#SI: 2
#L2: 1
#MAS: 10
