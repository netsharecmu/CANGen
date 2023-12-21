# rtf-tab, raw datasets
# bash run_small_scale.sh -p small-scale realtabformer-tabular -- openxc-nyc-downtown-east-no-imputation openxc-india-new-delhi-railway-to-aiims-no-imputation openxc-taiwan-highwayno2-can-no-imputation car-hacking-dos-hex car-hacking-fuzzy-hex car-hacking-rpm-hex car-hacking-gear-hex 
# syncan-raw

# rtf-tab, openxc/car-hacking processed datasets
# bash run_small_scale.sh -p small-scale realtabformer-tabular -- openxc-nyc-downtown-east openxc-india-new-delhi-railway-to-aiims openxc-taiwan-highwayno2-can car-hacking-dos-bits car-hacking-fuzzy-bits car-hacking-rpm-bits car-hacking-gear-bits

# ctgan, openxc processed datasets
# bash run_small_scale.sh -p small-scale ctgan -- openxc-nyc-downtown-east openxc-india-new-delhi-railway-to-aiims openxc-taiwan-highwayno2-can

# ctgan, car-hacking processed datasets
# bash run_small_scale.sh -p small-scale ctgan -- car-hacking-dos-bits car-hacking-fuzzy-bits car-hacking-rpm-bits car-hacking-gear-bits

# rtf-tab/ctgan, syncan-flag
# bash run_small_scale.sh -p small-scale realtabformer-tabular ctgan -- syncan-flag

# netshare, openxc sessionized
# bash run_small_scale.sh -p small-scale netshare -- openxc-nyc-downtown-east-sessionized openxc-india-new-delhi-railway-to-aiims-sessionized openxc-taiwan-highwayno2-can-sessionized

# netshare, car-hacking/syncan-flag-sessionized
# bash run_small_scale.sh -p small-scale netshare -- car-hacking-dos-bits-sessionized car-hacking-fuzzy-bits-sessionized car-hacking-rpm-bits-sessionized car-hacking-gear-bits-sessionized syncan-flag-sessionized

# bash run_small_scale.sh -p small-scale realtabformer-timeseries -- openxc-nyc-downtown-east-sessionized openxc-india-new-delhi-railway-to-aiims-sessionized openxc-taiwan-highwayno2-can-sessionized car-hacking-dos-bits-sessionized car-hacking-fuzzy-bits-sessionized car-hacking-rpm-bits-sessionized car-hacking-gear-bits-sessionized syncan-flag-sessionized

bash run_small_scale.sh -p small-scale tabddpm -- car-hacking-dos-bits car-hacking-fuzzy-bits car-hacking-rpm-bits car-hacking-gear-bits