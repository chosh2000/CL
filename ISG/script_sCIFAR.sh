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


# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.5 1 	   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho151_reg1_random.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho144_reg1_random1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--random_drop 1    | tee ${OUTDIR}/SIM_rho144_reg1_random2.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.7 0.7   --reglambda 1 --alpha 0.8	| tee ${OUTDIR}/SIM_rho177_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.6 0.6   --reglambda 1 --alpha 0.8	| tee ${OUTDIR}/SIM_rho166_reg1_a08.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.3 0.3   --reglambda 1 --alpha 1	| tee ${OUTDIR}/SIM_rho133_reg1_a1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.2 0.2   --reglambda 1 --alpha 1	| tee ${OUTDIR}/SIM_rho122_reg1_a1.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.4 1 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho141_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 1 0.4 --reglambda 1	| tee ${OUTDIR}/SIM_randomeven_rho114_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.5 0.5 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho155_reg1.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --dropmethod random_even --rho 1 0.5 1 --reglambda 1 	| tee ${OUTDIR}/SIM_randomeven_rho151_reg1.log

# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.12    | tee ${OUTDIR}/SIM_rho144_reg1_a12.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.14    | tee ${OUTDIR}/SIM_rho144_reg1_a14.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.16    | tee ${OUTDIR}/SIM_rho144_reg1_a16.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.18    | tee ${OUTDIR}/SIM_rho144_reg1_a18.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.20    | tee ${OUTDIR}/SIM_rho144_reg1_a20.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.22    | tee ${OUTDIR}/SIM_rho144_reg1_a22.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.24    | tee ${OUTDIR}/SIM_rho144_reg1_a24.log
# python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --out_dir $OUTDIR  --schedule 50 80  --rho 1 0.4 0.4   --reglambda 1 	--alpha 0.26    | tee ${OUTDIR}/SIM_rho144_reg1_a26.log


python -u ISG_sCIFAR.py --use_gpu True --repeat $REPEAT --init_model True --out_dir $OUTDIR  --schedule 50 80  --dropmethod rho --rho 1 0.5 0.5 --reglambda 1 --alpha 0	| tee ${OUTDIR}/SIM_rho155_reg1_dropout.log
