![tarel](assets/images/tarel_header.png)

# \[#\] Title

Temporal Adversarial Reconstruction of High Frequency Latent Spaces in Crypto Markets

# \[#\] Abstract

Latent Spaces are ...

# \[#\] Media

- Github : [TAReL](https://github.com/convergentist/TAReL)
- Arxiv (Pre-print) : [pending](https://github.com/convergentist/TAReL)
- HuggingFace Space : [pending](https://github.com/convergentist/TAReL)
- Papers-with-code : [pending](https://github.com/convergentist/TAReL)

# \[#\] Project structure

- Datasets: LOB - RP
- Models: w-GAN-gp
- Metrics: 

# \[#\] Run

## macOS

Create a virtualenv with pyenv and run the setup script

```bash
./scripts/ultradian_setup_venv.sh
```

```bash
pyenv shell m4_tarel_venv
source scripts/m4_tarel_env.sh
```


## Stage-One (Dataset Formation)

Logs from the experiment are in this folder `/artifacts/runs/00000000c001d00d/`,
the stage_one subfolder is about data preparation and analysis of recurrence
plots metrics. 

- Exchanges: Coinbase and Kraken
- Clocks: event, info_time, physical

Within the `buckets` folder, there is data about the effects of clock selection for 
the particular cell. In the `recurrence` sub-folder, the RQA metrics and 
other stats and analysis for the recurrence plots that were generated from each
of the enties or files within the buckets.

## Stage-Two (Model Training)

Logs from the model training exists in `stage_two` sub-folder, both for
the model training and testing.

- StabilityScoreSpec.json has
- Models are store here:  $CELL/$EXP/$CLOCK/checkpoints/step_/critic.pt and generator.pt

# \[#\] Cite

```bibtex
@misc{ref:ieee_guide,
  title        = {{IEEE} Reference Guide (V 11.12.2018)},
  organization = {IEEE Periodicals -- Transactions/Journal Department},
  address      = {Piscataway, NJ, USA},
  year         = {2018},
}
```

