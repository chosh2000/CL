OUTDIR=outputs/sCIFAR/$(date +'%b_%d_%Hh_%Mm')
REPEAT=1
mkdir -p $OUTDIR

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 0.5 	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho155_o1_a5.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 1 	--reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 1 1   	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_a5.log


# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1 	   --reglambda 0 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg0_a5.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1_a5.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 1 	--alpha 1    | tee ${OUTDIR}/SIM_rho111_reg1_a1.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--alpha 1    | tee ${OUTDIR}/SIM_rho151_reg1_a1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 1 	--alpha 1    | tee ${OUTDIR}/SIM_rho115_reg1_a1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 1 	--alpha 1    | tee ${OUTDIR}/SIM_rho155_reg1_a1.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--alpha 0.8    | tee ${OUTDIR}/SIM_rho151_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 1 	--alpha 0.8    | tee ${OUTDIR}/SIM_rho115_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 1 	--alpha 0.8    | tee ${OUTDIR}/SIM_rho155_reg1_a08.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--alpha 0.6    | tee ${OUTDIR}/SIM_rho151_reg1_a06.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 1 	--alpha 0.6    | tee ${OUTDIR}/SIM_rho115_reg1_a06.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 1 	--alpha 0.6    | tee ${OUTDIR}/SIM_rho155_reg1_a06.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--alpha 0.4    | tee ${OUTDIR}/SIM_rho151_reg1_a04.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 1 	--alpha 0.4    | tee ${OUTDIR}/SIM_rho115_reg1_a04.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 1 	--alpha 0.4    | tee ${OUTDIR}/SIM_rho155_reg1_a04.log

python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--alpha 0    | tee ${OUTDIR}/SIM_rho151_reg1_a0.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 1 	--alpha 0    | tee ${OUTDIR}/SIM_rho115_reg1_a0.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 1 	--alpha 0    | tee ${OUTDIR}/SIM_rho155_reg1_a0.log

