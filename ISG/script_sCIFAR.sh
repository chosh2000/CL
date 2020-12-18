OUTDIR=outputs/sCIFAR/$(date +'%b_%d_%Hh_%Mm')
REPEAT=1
mkdir -p $OUTDIR

python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100 120 --rho 0.5 0.5 0.5 --reglambda 1 --alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log
python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100 120 --rho 0.5 0.5 0.5 --reglambda 0.1 --alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100 120 --rho 0.5 0.5 0.5 --reglambda 1 --alpha 0.5    | tee ${OUTDIR}/SIM_rho555_reg1.log


# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100 120 --rho 1 0.5 0.5      --alpha 0.5    | tee ${OUTDIR}/SIM_rho155_a5.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 60 100 120 --rho 1 1 1          --alpha 0.5    | tee ${OUTDIR}/SIM_rho111_a5.log

