# StochasticTeacher-StudentKD

Welcome to the repository for "Learning from Stochastic Teacher Representations Using Student-Guided Knowledge Distillation." This README provides an overview of the project, developed as part of a 6-month internship, and outlines the key components and usage instructions.

## Context

**Professional Experience**
- **Position**: AI Research Engineer
- **Institution**: École de Technologie Supérieure (ETS) - Montréal, Canada
- **Duration**: April 2024 - September 2024
- **Laboratories**: Laboratoire d'Imagerie, Vision et Intelligence Artificielle (LIVIA) and International Laboratory on Learning Systems (ILLS)
- **Supervisor**: PhD Eric Granger

This research was conducted during a 6-month internship and culminated in a publication at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2025).

## Abstract

This project introduces a novel Stochastic Self-Distillation (SSD) training strategy designed to enhance deep learning model performance. By employing distillation-time dropout, the framework generates diverse teacher representations. The student model guides the knowledge distillation process, filtering and weighting task-relevant representations, resulting in state-of-the-art outcomes across various datasets without increasing model size or computational complexity. This approach is particularly suitable for resource-constrained applications such as wearable devices.

## Repository Structure

- `src/`: Main source code for implementing the model.
- `data/`: Datasets and data processing scripts.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `docs/`: Documentation and resources related to the project.
- `tests/`: Unit and integration tests.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train the model using the stochastic teacher framework, execute:

```bash
python src/train.py --config=configs/default.yaml
```

## Contributing

Contributions are welcome to improve or extend this research. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. Refer to the `LICENSE` file for more details.

## Acknowledgments

Special thanks to PhD Eric Granger and all team members at LIVIA and ILLS for their support and guidance during the internship. Additional thanks to the collaborators involved in this project.

---

For further details, please refer to the full [publication](https://www.researchgate.net/publication/390990429_Learning_from_Stochastic_Teacher_Representations_Using_Student-Guided_Knowledge_Distillation).
