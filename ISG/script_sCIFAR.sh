OUTDIR=outputs/sCIFAR/$(date +'%b_%d_%Hh_%Mm')
REPEAT=1
mkdir -p $OUTDIR

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 0.5 	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho155_o1_a5.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 0.5 1 	--reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100  --rho 1 1 1   	--reglambda 1   --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_a5.log


# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1 	   --reglambda 0 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg0_a5.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 0.5 0.5 0.5 --reglambda 1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 0.7 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg07_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 0.7 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg07_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 0.7 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho115_reg07_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 0.7 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho155_reg07_a5.log

python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 0.5  --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg05_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 0.5 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg05_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 0.5 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho115_reg05_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 0.5 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho155_reg05_a5.log


python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 0.3  --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg03_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 0.3 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg03_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 0.3 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho115_reg03_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 0.3 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho155_reg03_a5.log


python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 1   	   --reglambda 0.1  --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_reg01_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 0.1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho151_reg01_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 1 0.5 	   --reglambda 0.1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho115_reg01_a5.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 0.5   --reglambda 0.1 	--alpha 0.5    | tee ${OUTDIR}/SIM_rho155_reg01_a5.log


