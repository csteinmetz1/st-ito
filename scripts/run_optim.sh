CUDA_VISIBLE_DEVICES=1 python scripts/run_optim.py \
"/import/c4dm-datasets-ext/lcap-datasets/pst-benchmark-dataset/vocals/I_QWegHp-r0.wav" \
--target "/import/c4dm-datasets-ext/lcap-datasets/pst-benchmark-dataset/vocals/-o_MW5vifL8.wav" \
--algorithm es \
--effect-type vst \
--dropout 0.0 \
--max-iters 25 \
--savepop \
--metric param

# 
#["vocals/I_QWegHp-r0.wav", "vocals/-o_MW5vifL8.wav"],  # 1
#["vocals/n8cRTh4GEYg.wav", "vocals/CI2a5BxEIV0.wav"],  # 2
#["vocals/IyJ34F3tjG0.wav", "vocals/UGiEw22GI-4.wav"],  # 3
#["vocals/PGS0UvbCwGk.wav", "vocals/U1kifTk5xsU.wav"],  # 4
#["vocals/QP37fZmj-XY.wav", "vocals/CI2a5BxEIV0.wav"],  # 5
#["vocals/ScQISlpnjoQ.wav", "vocals/-o_MW5vifL8.wav"],  # 6
#["vocals/Slhrbuil8Yo.wav", "vocals/w1vxWWD1j50.wav"],  # 7
#["vocals/U1kifTk5xsU.wav", "vocals/w1vxWWD1j50.wav"],  # 8
#["vocals/UKyuxmgir2w.wav", "vocals/uOWK-ArhziU.wav"],  # 9
#["vocals/uOWK-ArhziU.wav", "vocals/Wbuj60Ew2p4.wav"],  # 10

#["guitar/q7dd3PAUpqE.wav", "guitar/1MxfbKkX7Zg.wav"],  # 1
#["guitar/q7dd3PAUpqE.wav", "guitar/5Az0vI2kU8o.wav"],  # 2
#["guitar/9uH5GvurJYc.wav", "guitar/8-lQhm67ZxE.wav"],  # 3
#["guitar/DPGanZQH6L4.wav", "guitar/8_tM8HPkR5w.wav"],  # 4
#["guitar/YDiUYW8gPbE.wav", "guitar/KqNrQw_Ne8w.wav"],  # 5
#["guitar/4cH_Q-uqJhU.wav", "guitar/7Mv-Et66FS4.wav"],  # 6
#["guitar/_xybjiuD9K0.wav", "guitar/DPGanZQH6L4.wav"],  # 7
#["guitar/MmUX2ZKhn_Q.wav", "guitar/KqNrQw_Ne8w.wav"],  # 8
#["guitar/BLrJSfrgYGI.wav", "guitar/ko8G5hkGqvc.wav"],  # 9
#["guitar/Fwnj5n1SdxY.wav", "guitar/wglmFyQPL4o.wav"],  # 10