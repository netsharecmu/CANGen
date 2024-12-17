# CANGen

[[Paper (ACSW' 24)](https://ieeexplore.ieee.org/abstract/document/10628782)]
[[Talk (ACSW' 24)]](https://acsw.unimore.it/2024/assets/videos/V06%20-%20Yin.mp4)

**Authors:** 
[[Yucheng Yin](https://sniperyyc.com/)]
[[Jorge Guajardo Merchan](https://www.bosch.com/de/forschung/ueber-bosch-research/unsere-experten/jorge-guajardo-merchan/)]
[[Pradeep Pappachan](https://www.linkedin.com/in/pradeep-pappachan-4a06051/)]
[[Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)]

**Abstract:**
Realistic CAN traces enable a wide range of automotive security applications including but not limited to anomaly detection, fingerprinting, simulation and testing. However, nowadays these applications are blocked by the sparsity of high-quality, diverse CAN traces. Building on the recent advances of Deep Generative Models (DGMs), we identify two key challenges when off-the-shelf DGMs are applied to CAN traces: generation and evaluation. We design an end-to-end, open-source framework CANGen which efficiently tackles the generation and evaluation of synthetic CAN traces with various DGMs. Our empirical evaluation shows that CANGen can efficiently generate high-fidelity synthetic CAN traces that meet the requirements of domain-specific properties and downstream tasks.

# Datasets and Preprocess
1. [OpenXC](http://openxcplatform.com/resources/traces.html)
    - NYC, Delhi, Taiwan
    - Example schema:
        ```Json
        {"name":"brake_pedal_status","value":true,"timestamp":1364310855.004000}
        {"name":"transmission_gear_position","value":"first","timestamp":1364310855.004000}
        {"name":"accelerator_pedal_position","value":0,"timestamp":1364323939.012000}
        {"name":"engine_speed","value":772,"timestamp":1364323939.027000}
        {"name":"vehicle_speed","value":0,"timestamp":1364323939.029000}
        {"name":"accelerator_pedal_position","value":0,"timestamp":1364323939.035000}
        {"name":"torque_at_transmission","value":3,"timestamp":1364323939.031000}
        ```
    - Preprocess (~3 mins): 
        1. select partial signals
        2. impute missing values
        3. convert to csv
        ```Bash
        cd preprocess/
        python3 openxc_json2csv.py
        ```
2. [ROAD](https://0xsam.com/road/)
    - Ambient (normal), attack
    - Example schema
        ```
        Label,Time,ID,Signal_1_of_ID,Signal_2_of_ID,Signal_3_of_ID,Signal_4_of_ID,Signal_5_of_ID,Signal_6_of_ID,Signal_7_of_ID,Signal_8_of_ID,Signal_9_of_ID,Signal_10_of_ID,Signal_11_of_ID,Signal_12_of_ID,Signal_13_of_ID,Signal_14_of_ID,Signal_15_of_ID,Signal_16_of_ID,Signal_17_of_ID,Signal_18_of_ID,Signal_19_of_ID,Signal_20_of_ID,Signal_21_of_ID,Signal_22_of_ID
        0,0.0,1413,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,
        0,1.9073486328125e-06,852,32765,0.0,120.0,11.0,,,,,,,,,,,,,,,,,,
        0,3.0994415283203125e-06,1505,511,1.0,3.0,1.0,0.0,32768.0,,,,,,,,,,,,,,,,
        ```
    - Preprocess (~10 min):
        1. Remove columns which are all NaN
        2. Impute missing values
        ```Bash
        cd preprocess/
        python3 road.py
        ```
    - *Note that constant columns are kept, e.g., `Labels` are all 0 for normal traces or some signal are constant.*

3. [OTIDS](https://sites.google.com/hksecurity.net/hcrl/Dataset/CAN-intrusion-dataset?pli=1)
    - Normal, attack
    - Example schema:
        ```
        Timestamp:          0.690329        ID: 0153    000    DLC: 8    b6 40 95 9c 71 15 68 82
        Timestamp:          0.691181        ID: 0164    000    DLC: 8    81 e6 b1 8b ea 8f cd d6
        Timestamp:          0.691743        ID: 05e4    000    DLC: 3    00 02 00
        Timestamp:          0.698796        ID: 02c0    100    DLC: 0
        ```
    - Preprocess (~10 min):
        1. Convert plaintext to column-based txt
        2. Replace non-exist data with NaN
        3. Convert hex strings (`ID`, `DATA_X`) to bit
        ```Bash
        cd preprocess/
        python3 otids_txt2csv.py
        ```

4. [Car-hacking](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
    - Normal, attack
    - Example schema: the same as OTIDS
    - Preprocess:
        - *Normal*: the same as OTIDS
        ```
        cd preprocess/
        python3 car_hacking_normal_txt2csv.py
        ```
        - *Attack*: the same as OTIDS
        ```
        cd preprocess/
        python3 car_hacking_attack_csv2csv.py
        ```

5. [Survival analysis](https://ocslab.hksecurity.net/Datasets/survival-ids)
    - Normal, attack
    - Example: the same as OTIDS
    - Preprocess:
        ```
        python survival_txt2csv.py
        ```

6. [Automotive CAN v2](https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2)
    - Normal, attack
    - Example:
        ```
        (1536574931.814734) slcan0 12A#00F0707200003090
        (1536574931.817030) slcan0 1F3#003C00
        (1536574931.817835) slcan0 0C1#80260BC683EB4399
        (1536574931.818045) slcan0 0C5#C0BFC74B80EBFD1B
        (1536574931.818296) slcan0 0D1#00000000000000
        (1536574931.818313) slcan0 185#1BA8
        ```
    - Preprocess:
        ```
        python3 automotive_can_v2_log2csv.py
        ```

7. [SynCAN](https://github.com/etas/SynCAN)
    - Normal, attack
    - Example: 
        ```
        Label,Time,ID,Signal1,Signal2,Signal3,Signal4
        0,2088.41338746,id5,0.0,0.9586862512831164
        0,2089.55410634,id8,0.24683876529406004
        0,2090.88561837,id3,0.2,1.0
        0,2091.65840611,id7,0.06388818948282671,0.0
        0,2100.36445933,id9,0.4495114006514658
        0,2100.47555051,id1,0.0,0.0
        ```
    - Preprocess:
        ```
        python3 syncan_csv2csv.py
        ```

8. [CarChallenge](https://ocslab.hksecurity.net/Datasets/carchallenge2020)
    - Normal, attack
    - Example:
        ```
        Timestamp,Arbitration_ID,DLC,Data,Class,SubClass
        1597759710.1258929,153,8,20 A1 10 FF 00 FF 50 1F,Normal,Normal
        1597759710.126151,220,8,13 24 7F 60 05 FF BF 10,Normal,Normal
        1597759710.1263099,507,4,08 00 00 01,Normal,Normal
        1597759710.127247,356,8,00 00 00 80 16 00 00 00,Normal,Normal
        1597759710.12748,340,8,FC 03 00 E4 B7 21 FA 3C,Normal,Normal
        1597759710.127698,366,7,33 B0 0A 33 30 00 01,Normal,Normal
        ```
    - Preprocess:
        ```
        python3 car_challenge_csv2csv.py
        ```

9. [TTIDS](https://github.com/EmbbededSecurity/AutomotiveSecurity/tree/main)
    - Normal, attack
    - Example:
        ```
        Timestamp,Arbitration_ID,DLC,Data,Class
        0.001476,381,8,80 08 40 00 00 84 B3 05,Normal
        0.001716,251,8,01 04 06 C0 00 01 08 80,Normal
        0.001917,2B0,6,80 FF 00 07 96 C9,Normal
        0.003657,47F,8,00 81 FF FA 00 78 00 07,Normal
        ```
    - Preprocess:
        ```
        python3 ttids_csv2csv.py
        ```

# Datasets used
- OpenXC
    - india_New_Delhi_Railway_to_AIIMS.csv
    - nyc_downtown_east.csv
    - taiwan_HighwayNo2_can.csv
- Car-hacking
    - All attack traces
    - Each attack trace is sampled to 1M samples, with train/test=0.7/0.3, i.e., train=700k, test=300k.
- SynCAN
    - train.csv (all normal by default)
    - Downsampled to 1M samples

- The preprocess of all other datasets may need to be adjusted accordingly (by the time of 11/29/2023).

# MISC
1. RTF seems to be problematic when running boostrapping with multi-process on raw datasets (with NaNs)... use single-process instead.