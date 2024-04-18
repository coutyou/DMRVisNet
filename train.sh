cd src && python train.py -p train --model-name DMRVisNet -c final --optim adam --gpu-id "0" --train-batch 16 --workers 8 --max-epochs 300 --lr 1e-5 --lambda-a 1 --lambda-t 1 --lambda-d 0.8 --lambda-defog 1e-6 --lambda-vis 1 --loss-a RMSE --loss-t RMSE --loss-d Reprojection --loss-defog MaskedRMSE --loss-vis MaskedL1 --t_thresh -1 --eps 1e-8 --height 288 --width 512