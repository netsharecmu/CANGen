# CANGen

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
        - *Normal*: the same as OTIDS
        ```
        cd preprocess/
        python3 survival_normal_txt2csv.py
        ```
        - *Attack*: the same as OTIDS
        ```
        cd preprocess/
        python3 survival_attack_txt2csv.py
        ```