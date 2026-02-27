## 批量xsens bvh处理
python scripts/xsens_bvh_to_robot_dataset.py --robot Q1 --src_folder motion_data/100STYLE/ --tgt_folder retargeting_data/Q1/100STYLE/ --override --num_cpus 16

## pkl播放
python general_motion_retargeting/utils/xsens_vendor/mujoco_retargeting_robot_view.py

## 批量pkl转csv
python general_motion_retargeting/utils/xsens_vendor/pkls_to_csvs.py --retargeting_data_folder retargeting_data/Q1/lafan_bvh/ --csv_folder lafan_Q1/lafan_bvh/

## 批量smplx处理
python scripts/smplx_to_robot_dataset.py --src_folder motion_data/AMASS/ --tgt_folder retargeting_data/Q1/AMASS/ --num_cpus 16 --override --robot Q1

