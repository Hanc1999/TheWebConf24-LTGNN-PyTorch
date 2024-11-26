for lr in 0.0005 0.0015 0.005 0.01
do
    for lamda in 1e-4 2e-4 5e-4 1e-3
    do
        for layer in 1
        do
            for batch in 2000 4000
            do
                for tune_index in 0 1 2
                do
                    python3 ./main.py --bpr_batch=$batch --decay=$lamda --lr=$lr --layer=$layer --seed=2020 --dataset="mba" --topks="[2, 5, 10, 20, 50, 100]" --recdim=64 --model="ltgnn" --appnp_alpha=0.45 --num_neighbors=15 --epoch=10 --device=0 --K_val=1 --LTGNN_selected_Ks="[1]" --tune_index=$tune_index
                done
            done
        done
    done
done