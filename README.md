# StochasticTeacher-StudentKD

Welcome to the repository for "Learning from Stochastic Teacher Representations Using Student-Guided Knowledge Distillation." This README provides an overview of the project and instructions for setting up and using the codebase.

## Context

**Professional Experience**
- **Position**: AI Research Intern
- **Institution**: École de Technologie Supérieure (ETS) - Montréal, Canada
- **Duration**: April 2024 - September 2024
- **Laboratories**: Laboratoire d'Imagerie, Vision et Intelligence Artificielle (LIVIA) and International Laboratory on Learning Systems (ILLS)
- **Supervisor**: PhD Eric Granger

This research was conducted during a 6-month internship and culminated in a publication at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2025).

## Abstract

This project introduces a novel Stochastic Self-Distillation (SSD) training strategy to enhance model performance. The framework uses distillation-time dropout to generate diverse teacher representations, allowing the student model to guide and weight task-relevant representations, achieving superior results without increasing model size or complexity.

## Directory Structure

```
SSD
|   LICENSE
|   README.md
|   requirements.txt
|
|   batch_distillation.sh
|   main_distillation.py
|       
\---src
    |   config.json
    |   logger_config.json
    |   ssd_utils.py
    |   parser.py
    |   Tdistillation.py
    |  __init__.py
    |   
    +---models
    |   |   main_painAttnNet.py
    |   |   module_mscn.py
    |   |   module_se_resnet.py
    |   |   module_transformer_encoder.py
    |   \   __init__.py
    |           
    +---tests
    |   |   test_generate_kfolds_index.py
    |   |   test_mscn.py
    |   |   test_PainAttnNet_output.py
    |   |   test_process_bioVid.py
    |   |   test_se_resnet.py
    |   |   test_transformer_encoder.py
    |   \   __init__.py
    |           
    +---trainers
    |   |   checkpoint_handler.py
    |   |   device_prep.py
    |   |   main_trainer.py
    |   |   metrics_manager.py
    |   \   __init__.py
    |           
    \---utils
        |   process_bioVid.py
        |   utils.py
        |   extractdistillresults.py
        \   __init__.py
```

## Teacher Training

Follow the instructions in the original PainAttentionNet Repository to set up and preprocess the dataset. Store the trained per-fold models for use as teacher models in student training. 

Link to PAN Repository: [PainAttnNet](https://github.com/zhenyuanlu/PainAttnNet/tree/main)

## Training Student Model

### Using Script
Run the following command:
```bash
sh batch_distillation.sh
```

### Using Python File
Run:
```bash
python main_distillation.py
```

Configure training settings in `config.json` and update required paths in all relevant files.

## Dataset

The dataset is available at the BioVid Heat Pain Database.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to PhD Eric Granger, LIVIA, ILLS, and all collaborators for their support during the internship.

---

For further details, refer to the full [publication](https://www.researchgate.net/publication/390990429_Learning_from_Stochastic_Teacher_Representations_Using_Student-Guided_Knowledge_Distillation).
